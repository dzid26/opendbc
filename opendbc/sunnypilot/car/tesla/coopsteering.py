"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
import numpy as np
from collections import namedtuple
from enum import Enum, auto

from opendbc.car import structs, rate_limit, DT_CTRL
from opendbc.car.vehicle_model import VehicleModel
from opendbc.sunnypilot.car import get_param
from opendbc.car.tesla.values import CarControllerParams

LKAS_OVERRIDE_OFF_SPEED = 6.0 # LKAS coop steering completely off below
LKAS_OVERRIDE_ON_SPEED = 7.0 # LKAS coop steering completely on above
LKAS_OVERRIDE_OFF_TORQUE = 1.3 # LKAS coop usually Off below this torque
LKAS_OVERRIDE_ON_TORQUE = 2.0 # LKAS coop usually On above this torque

STEER_OVERRIDE_LOW_SPEED_LO = LKAS_OVERRIDE_OFF_SPEED
STEER_OVERRIDE_LOW_SPEED_HI = LKAS_OVERRIDE_ON_SPEED

# angle override # todo implement steering torque inertia compensation to increase gains
STEER_OVERRIDE_MIN_TORQUE = 0.5 # Nm - based on typical steering bias + noise
STEER_OVERRIDE_MAX_TORQUE = 2.5 # Nm max torque before EPS disengages, LKAS takes over at 1.8Nm
STEER_OVERRIDE_MAX_LAT_ACCEL = 2.0 # m/s^2 - determines angle rate - speed dependent - similar to Tesla comfort steering mode
STEER_OVERRIDE_LAT_ACCEL_GAIN_LIMIT = 10 # deg/Nm stability and smoothness for angle control
# angle ramping
STEER_OVERRIDE_MAX_LAT_JERK = 2.0 # m/s^3 - determines angle ramping rate - speed dependent
STEER_OVERRIDE_MAX_LAT_JERK_CENTERING = CarControllerParams.ANGLE_LIMITS.MAX_LATERAL_JERK # m/s^3 -  for low speed angle ramp down
# stability and smoothness for angle ramp control - at very low speeds this takes precedence over jerk settings
STEER_OVERRIDE_LAT_JERK_GAIN_LIMIT = 150 # deg/s/Nm
STEER_OVERRIDE_TORQUE_RANGE = STEER_OVERRIDE_MAX_TORQUE - STEER_OVERRIDE_MIN_TORQUE

# model fighting mitigation
STEER_DESIRED_LIMITER_ALLOW_SPEED = LKAS_OVERRIDE_OFF_SPEED # m/s - below this speed the desired angle limiter is active
STEER_DESIRED_LIMITER_RATE_DELTA = 50 # deg/s/10ms when override angle ramp is active

# limit model acceleration when engaging or resuming from pause
STEER_RESUME_RATE_LIMIT_RAMP_RATE = 10 # deg/s/10ms - controls rate of rise of angle rate limit, not angle directly

# steer pause logic
STEER_PAUSE_ALLOW_SPEED = LKAS_OVERRIDE_ON_SPEED + 1.0 # enabling for higher speed can be dangerous if accidentally triggered
STEER_PAUSE_WAIT_TIME = 0.5 # s - wait time before disengaging after engagement with a stalk
STEER_PAUSE_HOLD_TARGET_DEVIATION = 10 # deg - wait time before disengaging after engagement with a stalk


CoopSteeringDataSP = namedtuple("CoopSteeringDataSP",
                                ["control_type", "lat_pause", "steeringAngleDeg"])


class CoopSteeringCarState:
  def __init__(self):
    pass


def get_steer_from_lat_accel(lat_accel, v_ego: float, VM: VehicleModel):
  """Calculate the maximum steering angle based on lateral acceleration."""
  curvature = lat_accel / (max(1, v_ego) ** 2)  # 1/m
  return math.degrees(VM.get_steer_from_curvature(curvature, v_ego, 0))  # deg


def apply_bounds(signal: float, limit: float) -> float:
  """Limit input to a range."""
  return np.clip(signal, -limit, limit)


def apply_deadzone(signal: float, deadzone: float) -> float:
  """Apply deadzone to input."""
  return signal - apply_bounds(signal, deadzone)


def calc_override_angle(torque: float, vEgo: float, VM: VehicleModel, lat_accel) -> float:
  """Map driver torque to lateral acceleration and convert to steering angle."""
  # lateral accel is linear in respect to angle so it's fine to interpolate it with torque
  torque_to_angle = get_steer_from_lat_accel(lat_accel, vEgo, VM) / STEER_OVERRIDE_TORQUE_RANGE

  # disable angle override below low speed # todo this could be removed after fixing stability issues
  gain = np.interp(vEgo, [STEER_OVERRIDE_LOW_SPEED_LO, STEER_OVERRIDE_LOW_SPEED_HI],
                         [0, STEER_OVERRIDE_LAT_ACCEL_GAIN_LIMIT])
  # limit the gain to prevent jerkiness and instability
  override_angle_target = torque * min(torque_to_angle, gain)

  return override_angle_target


def calc_override_angle_delta(torque: float, vEgo: float, VM: VehicleModel, lat_jerk) -> float:
  """Map driver torque to lateral jerk and convert to steering speed."""
  # prevents windup in carcontroller rate limiter
  lat_jerk = min(lat_jerk, CarControllerParams.ANGLE_LIMITS.MAX_LATERAL_JERK)

  # lateral accel is linear in respect to angle so it's fine to interpolate it with torque
  torque_to_angle = get_steer_from_lat_accel(lat_jerk, vEgo, VM) / STEER_OVERRIDE_TORQUE_RANGE
  # limit the gain to prevent jerkiness and instability
  override_angle_rate = torque * min(torque_to_angle, STEER_OVERRIDE_LAT_JERK_GAIN_LIMIT)

  # prevent windup due to carcontroller angle rate limiter
  return apply_bounds(override_angle_rate * DT_CTRL, CarControllerParams.ANGLE_LIMITS.MAX_ANGLE_RATE)


def lkas_compensation(apply_angle: float, apply_angle_last: float, steering_angle: float, driverTorque: float, vEgo: float) -> float:
  # lkas contribution is done by the car and is a difference between our command and measured angle
  lkas_angle = steering_angle - apply_angle_last
  # steering_angle can be lagging behind the command so ignore that:
  if driverTorque * lkas_angle < 0:
    lkas_angle = 0

  # smooth transition to LKAS based on enable torque
  lkas_angle = np.interp(abs(driverTorque),
                         [LKAS_OVERRIDE_OFF_TORQUE, LKAS_OVERRIDE_ON_TORQUE],
                         [0, lkas_angle])

  # get out of the way if below speed LKAS based torque blending
  if vEgo < LKAS_OVERRIDE_OFF_SPEED:
    lkas_angle = 0

  return apply_angle - lkas_angle


def get_lat_accel_from_steer(steer: float, v_ego: float, VM: VehicleModel):
  """Calculate the lateral acceleration based on steering angle."""
  curvature = VM.calc_curvature(math.radians(steer), v_ego, 0)  # 1/m
  return curvature * v_ego ** 2  # m/s^2


def est_holding_torque(steering_angle: float, vEgo: float, VM: VehicleModel):
  """Estimate torque necessary to hold steering wheel in place"""
  lat_accel = get_lat_accel_from_steer(steering_angle, vEgo, VM)
  torque = lat_accel / STEER_OVERRIDE_MAX_LAT_ACCEL * STEER_OVERRIDE_TORQUE_RANGE
  return torque


def override_above_holding_torque(driver_torque: float, holding_torque: float) -> bool:
  """
  Determines whether override torque is enough to hold the steering wheel in place (outward)
  Or if input is above min torque threshold (inward)
  """
  # if above max torque, then always determine the override
  holding_torque_limited = apply_bounds(holding_torque, STEER_OVERRIDE_MAX_TORQUE)

  if holding_torque_limited > 0: # same sign as CS.out.steeringAngleDeg
    torque_override_left = -STEER_OVERRIDE_MIN_TORQUE
    torque_override_right = max(holding_torque_limited, STEER_OVERRIDE_MIN_TORQUE)
  else:
    torque_override_left = min(holding_torque_limited, -STEER_OVERRIDE_MIN_TORQUE)
    torque_override_right = STEER_OVERRIDE_MIN_TORQUE

  return not (torque_override_left <= driver_torque <= torque_override_right)


class LateralPauseState(Enum):
  INIT_WAIT = auto()
  NORMAL = auto()
  PAUSE = auto()
  REENGAGE_HOLD_WAIT = auto()

  def __str__(self):
    return self.name


class PauseStateManager:
  def __init__(self):
    self.state = LateralPauseState.INIT_WAIT
    self.state_time = 0.0

  def reset_time(self):
    self.state_time = 0.0

  def update_state(self, new_state: LateralPauseState):
    """ Update the state of the pause state machine and reset timer. """
    if self.state != new_state:
      print(f"DEBUG: (Pause state) {self.state} -> {new_state}")
      self.state = new_state
      self.state_time = 0.0

  def tick(self, dt: float):
    """ Update the time spent in the current state. """
    self.state_time += dt

  def time_in_state(self) -> float:
    return self.state_time


class SteerRateLimiter:
  """Handles rate limiting of steering angle changes with a configurable rate."""
  def __init__(self):
    self.ramp_rate = 0
    self.apply_angle_last = 0.0

  def reset(self, apply_angle: float) -> None:
    """Reset the rate limiter state with the given angle."""
    self.apply_angle_last = apply_angle

  def update(self, apply_angle: float, angle_delta_lim: float) -> float:
    apply_angle_lim = rate_limit(apply_angle, self.apply_angle_last, -angle_delta_lim, angle_delta_lim)
    self.apply_angle_last = apply_angle_lim
    return apply_angle_lim


class CoopSteeringCarController:
  def __init__(self):
    super().__init__()
    self.coop_steering = CoopSteeringDataSP(False, False, 0)
    self.override_angle_accu = 0
    self.psm = PauseStateManager()
    self.resume_rate_limiter_delta = SteerRateLimiter()
    self.resume_rate_limiter = SteerRateLimiter()
    self.override_accel_rate_limiter_delta = SteerRateLimiter()
    self.override_accel_rate_limiter = SteerRateLimiter()

  def reset_pause_state(self):
    self.psm.update_state(LateralPauseState.INIT_WAIT)
    self.psm.reset_time()

  def update_pause_state(self, angle_planned: float, CS: structs.CarState, VM: VehicleModel):
    torque_hold = est_holding_torque(CS.out.steeringAngleDeg, CS.out.vEgoRaw, VM)
    torque_override = override_above_holding_torque(CS.out.steeringTorque, torque_hold)

    # Engage conditions (when to enter or stay in LATERAL_PAUSED)
    should_disengage = (
      CS.out.vEgoRaw < STEER_PAUSE_ALLOW_SPEED # todo add hysteresis
      and self.psm.time_in_state() > STEER_PAUSE_WAIT_TIME
      and CS.out.steeringPressed
      and torque_override # todo add small debounce
    )

    # Reengage conditions when hands released steering wheel
    should_reengage_released = (
      not CS.out.steeringPressed
      # and not CS.out.standstill
      and not should_disengage
    )

    # Reengage conditions when hands holding steering wheel at desired controls angle for some time
    is_far_from_target = abs(CS.out.steeringAngleDeg - angle_planned) > STEER_PAUSE_HOLD_TARGET_DEVIATION
    should_hold_wait = (
      CS.out.steeringRateDeg == 0
      and CS.hands_on_level < 3
      and not is_far_from_target
      # and not CS.out.standstill
    )
    should_reengage_from_hold = (self.psm.time_in_state() > 1.0 or should_reengage_released) \
      and not should_disengage

    # State transitions
    if self.psm.state == LateralPauseState.INIT_WAIT and self.psm.time_in_state() > STEER_PAUSE_WAIT_TIME:
      self.psm.update_state(LateralPauseState.NORMAL)
    elif self.psm.state == LateralPauseState.NORMAL and should_disengage:
      self.psm.update_state(LateralPauseState.PAUSE)
    elif self.psm.state == LateralPauseState.PAUSE and should_reengage_released:
      self.psm.update_state(LateralPauseState.NORMAL)
    elif self.psm.state == LateralPauseState.PAUSE and should_hold_wait:
      self.psm.update_state(LateralPauseState.REENGAGE_HOLD_WAIT)
    elif self.psm.state == LateralPauseState.REENGAGE_HOLD_WAIT and not should_hold_wait:
      self.psm.update_state(LateralPauseState.PAUSE)
    elif self.psm.state == LateralPauseState.REENGAGE_HOLD_WAIT and should_reengage_from_hold:
      self.psm.update_state(LateralPauseState.NORMAL)

    self.psm.tick(DT_CTRL)

    lat_pause = self.psm.state == LateralPauseState.PAUSE \
        or self.psm.state == LateralPauseState.REENGAGE_HOLD_WAIT
    return lat_pause

  def apply_override_angle(self, lat_active: bool, apply_angle: float, driverTorque: float, vEgo: float, VM: VehicleModel) -> float:
    """
    Emulates steering springiness based on lateral acceleration exerted on the steering rack.
    At low speed the max angle approaches infinity, so the conversion torque to angle has to be limited (STEER_OVERRIDE_LAT_ACCEL_GAIN_LIMIT).
    We rely on apply_override_angle_ramp to reach the max angle at low speeds.
    """
    if not lat_active:
      return apply_angle

    ## torque to position
    # ignore torque sensor offset and disturbances
    steering_torque_with_deadzone = apply_deadzone(driverTorque, STEER_OVERRIDE_MIN_TORQUE)
    angle_override = calc_override_angle(steering_torque_with_deadzone, vEgo, VM, STEER_OVERRIDE_MAX_LAT_ACCEL)
    return apply_angle + angle_override

  def apply_override_angle_ramp(self, lat_active: bool, lkas_enabled: bool, apply_angle: float, driverTorque: float, vEgo: float, VM: VehicleModel) -> float:
    """
    Emulates steering rotation for low speed when steering resistance due to lateral acceleration is not well defined.
    Physically torque to angle rate corresponds to viscous damping of the wheels on the ground.
    However here lateral jerk limit is used as a proxy for estimating reasonable safe steering angle rate depending on the vehicle speed.
    Ramp maximum angle is limited according to lateral acceleration limits.
    """
    if not lat_active:
      self.override_angle_accu = 0
      return apply_angle

    # disable ramping at high speed -
    # prevents large slow swings due to LKAS reducing input resistance when target is off center;
    # limits torque to minimum so it allows ramping down the angle if already extended
    if lkas_enabled:
      driverTorque = np.interp(vEgo, [LKAS_OVERRIDE_OFF_SPEED, LKAS_OVERRIDE_ON_SPEED],
              [driverTorque, apply_bounds(driverTorque, STEER_OVERRIDE_MIN_TORQUE)])

    # torque biasing emulates the steering centering:
    if self.override_angle_accu > 0:
      torque_biased = driverTorque - STEER_OVERRIDE_MIN_TORQUE
    elif self.override_angle_accu < 0:
      torque_biased = driverTorque + STEER_OVERRIDE_MIN_TORQUE
    else:
      torque_biased = apply_deadzone(driverTorque, STEER_OVERRIDE_MIN_TORQUE)

    # higher rate when centering
    angle_override_delta = calc_override_angle_delta(torque_biased, vEgo, VM,
                          STEER_OVERRIDE_MAX_LAT_JERK if torque_biased * self.override_angle_accu > 0
                          else STEER_OVERRIDE_MAX_LAT_JERK_CENTERING)

    # ramp the angle
    new_override_angle_accu = self.override_angle_accu + angle_override_delta
    # clamp to 0 if sign changes
    if new_override_angle_accu * self.override_angle_accu < 0 and abs(driverTorque) < STEER_OVERRIDE_MIN_TORQUE:
      self.override_angle_accu = 0
    else:
      self.override_angle_accu = new_override_angle_accu

    # steering should rotate until reaches angle set by the desired max lat accel
    self.override_angle_accu = apply_bounds(self.override_angle_accu,
                                            get_steer_from_lat_accel(STEER_OVERRIDE_MAX_LAT_ACCEL, vEgo, VM))

    apply_angle = apply_angle + self.override_angle_accu

    # predict and prevent windup due to Carcontroller angle saturation
    absolute_max_angle = min(CarControllerParams.ANGLE_LIMITS.STEER_ANGLE_MAX, # 360deg
                          get_steer_from_lat_accel(CarControllerParams.ANGLE_LIMITS.MAX_LATERAL_ACCEL, vEgo, VM))
    angle_saturation_delta = apply_angle - apply_bounds(apply_angle, absolute_max_angle)
    self.override_angle_accu -= angle_saturation_delta
    return apply_angle

  def steer_desired_accel_limit_for_override(self, lat_active: bool, apply_angle: float, steering_angle: float, override_active: bool) -> float:
    """Acceleration limit angle override is active"""
    if not lat_active:
      self.override_accel_rate_limiter_delta.reset(0)
      self.override_accel_rate_limiter.reset(apply_angle)
      return apply_angle

    if override_active:
      max_angle_rate_delta = STEER_DESIRED_LIMITER_RATE_DELTA
    else:
      max_angle_rate_delta = CarControllerParams.ANGLE_LIMITS.MAX_ANGLE_RATE / DT_CTRL
    angle_delta = apply_angle - self.override_accel_rate_limiter.apply_angle_last
    angle_delta_new = self.override_accel_rate_limiter_delta.update(angle_delta, max_angle_rate_delta * DT_CTRL)
    apply_angle = self.override_accel_rate_limiter.update(apply_angle, angle_delta_new)

    return apply_angle

  def resume_steer_rate_limit_ramp(self, lat_active: bool, apply_angle: float, steering_angle: float) -> float:
    """Limits steering wheel acceleration when resuming steering after pause"""
    if not lat_active:
      # reset and bypass when paused
      self.resume_rate_limiter_delta.reset(0)
      self.resume_rate_limiter.reset(steering_angle)
      return apply_angle

    angle_rate_delta_lim = self.resume_rate_limiter_delta.update(CarControllerParams.ANGLE_LIMITS.MAX_LATERAL_JERK,
                                                         STEER_RESUME_RATE_LIMIT_RAMP_RATE * DT_CTRL)
    apply_angle_lim = self.resume_rate_limiter.update(apply_angle, angle_rate_delta_lim)
    return apply_angle_lim

  def coop_steering_update(self, CC: structs.CarControl, CC_SP: structs.CarControlSP, CS: structs.CarState, VM: VehicleModel) -> CoopSteeringDataSP:
    lkas_enabled = get_param(CC_SP.params, "TeslaLkasSteering", False)
    angle_coop_enabled = get_param(CC_SP.params, "TeslaCoopSteering", False)
    low_speed_pause_enabled = get_param(CC_SP.params, "TeslaLowSpeedSteerPause", False)

    # Replicate carcontroller behavior here to perform actions on all disengagements
    lat_active = CC.latActive and CS.hands_on_level < 3
    apply_angle = CC.actuators.steeringAngleDeg

    # 1 = angle control, 2 = LKAS mode; todo: use CAN parser enums
    control_type = 2 if lkas_enabled else 1

    if low_speed_pause_enabled and lat_active:
      lat_pause = self.update_pause_state(apply_angle, CS, VM)
    else:
      self.reset_pause_state()
      lat_pause = False

    lat_active = lat_active and not lat_pause

    # avoid sudden rotation on engagement
    apply_angle = self.resume_steer_rate_limit_ramp(lat_active, apply_angle, CS.out.steeringAngleDeg)

    if angle_coop_enabled:
      apply_angle = self.steer_desired_accel_limit_for_override(lat_active, apply_angle, CS.out.steeringAngleDeg,
                                        self.override_angle_accu != 0 and CS.out.vEgo < STEER_DESIRED_LIMITER_ALLOW_SPEED)
      apply_angle = self.apply_override_angle(lat_active, apply_angle, CS.out.steeringTorque, CS.out.vEgo, VM)
      if not low_speed_pause_enabled:
        # todo maybe keep it always enabled at high speed for consistent behavior
        apply_angle = self.apply_override_angle_ramp(lat_active, lkas_enabled, apply_angle, CS.out.steeringTorque, CS.out.vEgo, VM)

      if lkas_enabled:  # apply LKAS compensation to angle override
        apply_angle = lkas_compensation(apply_angle, self.coop_steering.steeringAngleDeg, CS.out.steeringAngleDeg,
                                        CS.out.steeringTorque, CS.out.vEgo)

    self.coop_steering = CoopSteeringDataSP(control_type, lat_pause, apply_angle)
    return self.coop_steering

  def update(self, CC: structs.CarControl, CC_SP: structs.CarControlSP, CS: structs.CarState) -> CoopSteeringDataSP:
    coop_steering = self.coop_steering_update(CC, CC_SP, CS, self.VM)

    return coop_steering
