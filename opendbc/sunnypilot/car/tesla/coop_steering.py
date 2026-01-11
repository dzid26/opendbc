"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
import numpy as np
from collections import namedtuple
from dataclasses import replace

from opendbc.car import structs, rate_limit, DT_CTRL
from opendbc.car.vehicle_model import VehicleModel
from opendbc.car.lateral import apply_steer_angle_limits_vm
from opendbc.car.tesla.values import CarControllerParams
from opendbc.sunnypilot.car.tesla.values import TeslaFlagsSP
from opendbc.sunnypilot.car.tesla.steer_pause import PauseManager


DT_LAT_CTRL = DT_CTRL * CarControllerParams.STEER_STEP

class CoopSteeringCarControllerParams(CarControllerParams):
  ANGLE_LIMITS = replace(CarControllerParams.ANGLE_LIMITS, MAX_ANGLE_RATE=5)

STEERING_DEG_PHASE_LEAD_COEFF = 8.0

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
STEER_OVERRIDE_MAX_LAT_JERK_CENTERING = CoopSteeringCarControllerParams.ANGLE_LIMITS.MAX_LATERAL_JERK # m/s^3 -  for low speed angle ramp down
# stability and smoothness for angle ramp control - at very low speeds this takes precedence over jerk settings
STEER_OVERRIDE_LAT_JERK_GAIN_LIMIT = 150 # deg/s/Nm
STEER_OVERRIDE_TORQUE_RANGE = STEER_OVERRIDE_MAX_TORQUE - STEER_OVERRIDE_MIN_TORQUE

# model fighting mitigation
STEER_DESIRED_LIMITER_ALLOW_SPEED = LKAS_OVERRIDE_OFF_SPEED # m/s - below this speed the desired angle limiter is active
STEER_DESIRED_LIMITER_ACCEL = 100 # deg/s^2 when override angle ramp is active
STEER_DESIRED_LIMITER_OVERRIDE_ACTIVE_COUNTER = 3.0 # second

# limit model acceleration when engaging or resuming from pause
STEER_RESUME_RATE_LIMIT_RAMP_RATE = 500 # deg/s^2 - controls rate of rise of angle rate limit, not angle directly


CoopSteeringDataSP = namedtuple("CoopSteeringDataSP",
                                ["steeringAngleDeg", "lat_active", "control_type"])


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
  lat_jerk = min(lat_jerk, CoopSteeringCarControllerParams.ANGLE_LIMITS.MAX_LATERAL_JERK)

  # lateral accel is linear in respect to angle so it's fine to interpolate it with torque
  torque_to_angle = get_steer_from_lat_accel(lat_jerk, vEgo, VM) / STEER_OVERRIDE_TORQUE_RANGE
  # limit the gain to prevent jerkiness and instability
  override_angle_rate = torque * min(torque_to_angle, STEER_OVERRIDE_LAT_JERK_GAIN_LIMIT)

  # prevent windup in angle rate limiter
  return apply_bounds(override_angle_rate * DT_LAT_CTRL, CoopSteeringCarControllerParams.ANGLE_LIMITS.MAX_ANGLE_RATE)


def lkas_compensation(apply_angle: float, apply_angle_final_last: float, steering_angle: float, driverTorque: float, vEgo: float) -> float:
  # lkas contribution is done by the car and is a difference between our command and measured angle
  lkas_angle = steering_angle - apply_angle_final_last
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


class SteerRateLimiter:
  """Handles rate limiting of steering angle changes with a configurable rate."""
  def __init__(self):
    self._last = 0.0

  def reset(self, angle: float) -> None:
    """Reset the rate limiter state with the given angle."""
    self._last = angle

  def update(self, angle: float, angle_delta_lim: float) -> float:
    angle_lim = rate_limit(angle, self._last, -angle_delta_lim, angle_delta_lim)
    self._last = angle_lim
    return angle_lim


class AngleSlew:
  angle_cmd: float = 0.0
  vel_cmd: float = 0.0
  acc_cmd: float = 0.0

  def reset(self, angle: float = 0.0) -> None:
    self.angle_cmd = angle
    self.vel_cmd = 0.0
    self.acc_cmd = 0.0

  def update(
      self,
      target_angle: float,
      dt: float,
      v_max: float,
      a_max: float,
      j_max: float,
      eps: float = 1e-8,
  ) -> float:
    """
    Advance the command by dt toward target_angle subject to limits.

    Args:
      target_angle: desired angle (same units as angle_cmd)
      dt: timestep (s)
      v_max: max velocity magnitude
      a_max: max acceleration magnitude
      j_max: max jerk magnitude (rate of change of acceleration)
      eps: small tolerance to treat as zero
    Returns:
      The updated angle command.
    """
    err = target_angle - self.angle_cmd

    if abs(err) < eps and abs(self.vel_cmd) < eps:
      self.angle_cmd = target_angle
      self.vel_cmd = 0.0
      self.acc_cmd = 0.0
      return self.angle_cmd

    # braking distance (always >= 0)
    bd = (self.vel_cmd * self.vel_cmd) / (2.0 * max(a_max, eps))

    # pick desired acceleration sign
    if abs(err) <= bd + 1e-12:
      # brake
      if abs(self.vel_cmd) > 1e-12:
        desired_acc = -a_max if self.vel_cmd > 0 else a_max
      else:
        desired_acc = 0.0
    else:
      # accelerate toward target
      desired_acc = a_max if err > 0 else -a_max

    # jerk limit
    max_acc_delta = j_max * dt
    desired_acc = np.clip(desired_acc, self.acc_cmd - max_acc_delta, self.acc_cmd + max_acc_delta)

    # integrate
    self.acc_cmd = desired_acc
    self.vel_cmd += self.acc_cmd * dt
    self.vel_cmd = np.clip(self.vel_cmd, -v_max, v_max)
    self.angle_cmd += self.vel_cmd * dt

    # prevent crossing/overshoot
    new_err = target_angle - self.angle_cmd
    if not np.isclose(err, 0.0, atol=0.0):
      crossed = (err > 0 and new_err <= 0) or (err < 0 and new_err >= 0)
      if crossed:
        self.angle_cmd = target_angle
        self.vel_cmd = 0.0
        self.acc_cmd = 0.0

    return self.angle_cmd


class CoopSteeringCarController:
  def __init__(self):
    self.coop_steeringAngleDeg = 0
    self.override_angle_accu = 0
    self.override_active_counter = 0  # Counter for how many cycles torque is below threshold
    self.pause_manager = PauseManager()
    self.resume_rate_limiter_delta = SteerRateLimiter()
    self.resume_rate_limiter = SteerRateLimiter()
    self.override_accel_rate_limiter = AngleSlew()
    self.debug_angle_desired_limited = 0

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

    # torque biasing emulates the steering centering when released:
    if self.override_angle_accu > 0 and abs(vEgo) > 0.1:
      torque_biased = driverTorque - STEER_OVERRIDE_MIN_TORQUE
    elif self.override_angle_accu < 0 and abs(vEgo) > 0.1:
      torque_biased = driverTorque + STEER_OVERRIDE_MIN_TORQUE
    else:
      torque_biased = apply_deadzone(driverTorque, STEER_OVERRIDE_MIN_TORQUE)

    # higher rate when centering
    angle_override_delta = calc_override_angle_delta(torque_biased, vEgo, VM,
                          STEER_OVERRIDE_MAX_LAT_JERK if (torque_biased * self.override_angle_accu) > 0
                          else STEER_OVERRIDE_MAX_LAT_JERK_CENTERING)

    # ramp the angle
    new_override_angle_accu = self.override_angle_accu + angle_override_delta
    # snap to 0 if sign changes
    if (new_override_angle_accu * self.override_angle_accu) < 0 and abs(driverTorque) < STEER_OVERRIDE_MIN_TORQUE:
      self.override_angle_accu = 0
    else:
      self.override_angle_accu = new_override_angle_accu

    # steering should rotate until reaches angle set by the desired max lat accel
    self.override_angle_accu = apply_bounds(self.override_angle_accu,
                                            min(get_steer_from_lat_accel(STEER_OVERRIDE_MAX_LAT_ACCEL, vEgo, VM),
                                                CoopSteeringCarControllerParams.ANGLE_LIMITS.STEER_ANGLE_MAX))

    return self.override_angle_accu + apply_angle

  def overriding_steer_desired_accel_limit(self, lat_active: bool, apply_angle: float, vEgo: float, steeringTorque: float) -> float:
    """
    Acceleration rate limiter - limits acceleration but allows for quick deceleration (no overshoot)
    """
    if not lat_active:
      self.override_accel_rate_limiter.reset(apply_angle)
      return apply_angle

    if abs(steeringTorque) >= STEER_OVERRIDE_MIN_TORQUE:
      self.override_active_counter = 0
    else:
      self.override_active_counter += DT_LAT_CTRL
      self.override_active_counter = min(self.override_active_counter, STEER_DESIRED_LIMITER_OVERRIDE_ACTIVE_COUNTER)

    max_angle_rate = CarControllerParams.ANGLE_LIMITS.MAX_ANGLE_RATE / DT_LAT_CTRL # MAX_ANGLE_RATE is per frame units so convert to real rate
    # this ensures no acceleration limit when override is disabled:
    max_angle_accel = max_angle_rate / DT_LAT_CTRL # ensures max deceleration
    if vEgo < STEER_DESIRED_LIMITER_ALLOW_SPEED:
      # Interpolate between STEER_DESIRED_LIMITER_ACCEL and max_angle_accel based on counter progress
      max_angle_accel = np.interp(
        self.override_active_counter,
        [0, STEER_DESIRED_LIMITER_OVERRIDE_ACTIVE_COUNTER],
        [STEER_DESIRED_LIMITER_ACCEL, max_angle_accel]
      )
    return self.override_accel_rate_limiter.update(apply_angle, DT_LAT_CTRL, max_angle_rate, max_angle_accel, 100_000.0) #, snap = False)

  def resume_steer_desired_rate_limit(self, lat_active: bool, apply_angle: float, steering_angle: float) -> float:
    """Limits steering wheel acceleration when resuming steering after pause"""
    if not lat_active:
      # reset and bypass when paused
      self.resume_rate_limiter_delta.reset(0)
      self.resume_rate_limiter.reset(steering_angle)
      return steering_angle

    angle_rate_delta_lim = self.resume_rate_limiter_delta.update(CarControllerParams.ANGLE_LIMITS.MAX_ANGLE_RATE,
                                                         STEER_RESUME_RATE_LIMIT_RAMP_RATE * DT_LAT_CTRL**2)
    apply_angle_lim = self.resume_rate_limiter.update(apply_angle, angle_rate_delta_lim)
    return apply_angle_lim

  def coop_steering_update(self, apply_angle, lat_active, CP_SP: structs.CarParamsSP, CS: structs.CarState, VM: VehicleModel) -> CoopSteeringDataSP:
    # estimate real steering angle by adding rate to the tesla filtered angle
    steeringAngleDegPhaseLead = CS.out.steeringAngleDeg + CS.out.steeringRateDeg / STEERING_DEG_PHASE_LEAD_COEFF

    lkas_enabled = CP_SP.flags & TeslaFlagsSP.LKAS_STEERING.value
    angle_coop_enabled = CP_SP.flags & TeslaFlagsSP.COOP_STEERING.value
    low_speed_pause_enabled = CP_SP.flags & TeslaFlagsSP.PAUSE_STEERING.value

    # 1 = angle control, 2 = LKAS mode; todo: use CAN parser enums
    control_type = 2 if lkas_enabled else 1

    if low_speed_pause_enabled and lat_active:
      lat_pause = self.pause_manager.update_pause_state(apply_angle, CS, VM)
    else:
      self.pause_manager.reset_pause_state()
      lat_pause = False

    lat_active = lat_active and not lat_pause

    # avoid sudden rotation on engagement
    apply_angle = self.resume_steer_desired_rate_limit(lat_active, apply_angle, steeringAngleDegPhaseLead)

    if angle_coop_enabled:
      apply_angle = self.overriding_steer_desired_accel_limit(lat_active, apply_angle, CS.out.vEgo, CS.out.steeringTorque)
      self.debug_angle_desired_limited = apply_angle #! debug
      apply_angle = self.apply_override_angle(lat_active, apply_angle, CS.out.steeringTorque, CS.out.vEgo, VM)
      if not low_speed_pause_enabled:
        # todo maybe keep it always enabled at high speed for consistent behavior
        apply_angle = self.apply_override_angle_ramp(lat_active, lkas_enabled, apply_angle, CS.out.steeringTorque, CS.out.vEgo, VM)

      if lkas_enabled:  # apply LKAS compensation to angle override
        apply_angle = lkas_compensation(apply_angle, self.coop_steeringAngleDeg, steeringAngleDegPhaseLead,
                                        CS.out.steeringTorque, CS.out.vEgo)

    # final rate limit - matching panda safety
    self.coop_steeringAngleDeg = apply_steer_angle_limits_vm(apply_angle, self.coop_steeringAngleDeg, CS.out.vEgoRaw,
                                                    CS.out.steeringAngleDeg, lat_active, CoopSteeringCarControllerParams, VM)

    return CoopSteeringDataSP(self.coop_steeringAngleDeg, lat_active, control_type)

  def update(self, apply_angle, lat_active, CP_SP: structs.CarParamsSP, CS: structs.CarState) -> CoopSteeringDataSP:
    return self.coop_steering_update(apply_angle, lat_active, CP_SP, CS, self.VM)
