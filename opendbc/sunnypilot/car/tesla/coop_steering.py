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


DT_LAT_CTRL = DT_CTRL * CarControllerParams.STEER_STEP

class CoopSteeringCarControllerParams(CarControllerParams):
  ANGLE_LIMITS = replace(CarControllerParams.ANGLE_LIMITS, MAX_ANGLE_RATE=5)

STEERING_DEG_PHASE_LEAD_COEFF = 8.0

# angle override # todo implement steering torque inertia compensation to increase gains
STEER_OVERRIDE_MIN_TORQUE = 0.5 # Nm - based on typical steering bias + noise
STEER_OVERRIDE_MAX_TORQUE = 2.5 # Nm max torque before EPS disengages
STEER_OVERRIDE_MAX_LAT_ACCEL = 1.5 # m/s^2 - determines angle rate - speed dependent - similar to Tesla comfort steering mode
STEER_OVERRIDE_LAT_ACCEL_GAIN_LIMIT = 10 # deg/Nm stability and smoothness for angle control  # todo this could be increased after solving feedback stability

# angle ramping
STEER_OVERRIDE_MAX_LAT_JERK = 2.0 # m/s^3 - determines angle ramping rate - speed dependent
STEER_OVERRIDE_MAX_LAT_JERK_CENTERING = CoopSteeringCarControllerParams.ANGLE_LIMITS.MAX_LATERAL_JERK # m/s^3 -  for low speed angle ramp down
# stability and smoothness for angle ramp control - at very low speeds this takes precedence over jerk settings
STEER_OVERRIDE_TORQUE_RANGE = STEER_OVERRIDE_MAX_TORQUE - STEER_OVERRIDE_MIN_TORQUE
STEER_OVERRIDE_LAT_JERK_GAIN_LIMIT = 100 # deg/s/Nm - should be less than CarControllerParams.ANGLE_LIMITS.MAX_ANGLE_RATE/DT_CTRL/STEER_OVERRIDE_TORQUE_RANGE


# limit model acceleration when engaging
STEER_RESUME_RATE_LIMIT_RAMP_RATE = 500 # deg/s^2 - controls rate of rise of angle rate limit, not angle directly


CoopSteeringDataSP = namedtuple("CoopSteeringDataSP",
                                ["steeringAngleDeg", "lat_active"])

def get_steer_from_lat_accel(lat_accel, v_ego: float, VM: VehicleModel):
  """Calculate the maximum steering angle based on lateral acceleration."""
  curvature = lat_accel / (max(1, v_ego) ** 2)  # 1/m
  return math.degrees(VM.get_steer_from_curvature(curvature, v_ego, 0))  # deg


def apply_bounds(signal: float, limit: float) -> float:
  """Limit input to a range."""
  return float(np.clip(signal, -limit, limit))


def apply_deadzone(signal: float, deadzone: float) -> float:
  """Apply deadzone to input."""
  return signal - apply_bounds(signal, deadzone)


def calc_override_angle_limited(torque: float, vEgo: float, VM: VehicleModel, lat_accel) -> float:
  """
  Map driver torque to lateral acceleration and convert to steering angle.
  Limit gain for stability with EPS and torque sensor interaction.
  """

  # lateral accel is linear in respect to angle so it's fine to interpolate it with torque
  torque_to_angle = get_steer_from_lat_accel(lat_accel, vEgo, VM) / STEER_OVERRIDE_TORQUE_RANGE

  # limit the gain to prevent jerkiness and instability
  gain_limit = STEER_OVERRIDE_LAT_ACCEL_GAIN_LIMIT
  override_angle_target = torque * min(torque_to_angle, gain_limit)

  return override_angle_target


def calc_override_angle_delta_limited(torque: float, vEgo: float, VM: VehicleModel, lat_jerk) -> float:
  """
  Map driver torque to lateral jerk and convert to steering speed.
  Limit gain for stability with EPS and torque sensor interaction.
  """

  # prevents windup in carcontroller rate limiter
  lat_jerk = min(lat_jerk, CoopSteeringCarControllerParams.ANGLE_LIMITS.MAX_LATERAL_JERK)

  # lateral accel is linear in respect to angle so it's fine to interpolate it with torque
  torque_to_angle = get_steer_from_lat_accel(lat_jerk, vEgo, VM) / STEER_OVERRIDE_TORQUE_RANGE
  # limit the gain to prevent jerkiness and instability
  gain_limit = min(STEER_OVERRIDE_LAT_JERK_GAIN_LIMIT, CarControllerParams.ANGLE_LIMITS.MAX_ANGLE_RATE / DT_CTRL / STEER_OVERRIDE_TORQUE_RANGE)
  override_angle_rate = torque * min(torque_to_angle, gain_limit)

  # prevent windup in angle rate limiter
  return apply_bounds(override_angle_rate * DT_LAT_CTRL, CoopSteeringCarControllerParams.ANGLE_LIMITS.MAX_ANGLE_RATE)


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


class CoopSteeringCarController:
  def __init__(self):
    self.coop_apply_angle_last = 0
    self.coop_apply_angle_last_sat = 0
    self.override_angle_accu = 0
    self.resume_rate_limiter_delta = SteerRateLimiter()
    self.resume_rate_limiter = SteerRateLimiter()
    self.debug_angle_desired_limited = 0

  def apply_override_angle_direct(self, lat_active: bool, driverTorque: float, vEgo: float, VM: VehicleModel) -> float:
    """
    Emulates steering springiness based on lateral acceleration exerted on the steering rack.
    We rely on apply_override_angle_ramp to reach the max angle at low speeds.
    At low speed lateral acceleration approaches infinity and it is not good proxy
    for the torque to target angle conversion and needs to be limited

    """
    if not lat_active:
      return 0.0

    ## torque to position
    # ignore torque sensor offset and disturbances
    steering_torque_with_deadzone = apply_deadzone(driverTorque, STEER_OVERRIDE_MIN_TORQUE)
    angle_override = calc_override_angle_limited(steering_torque_with_deadzone, vEgo, VM, STEER_OVERRIDE_MAX_LAT_ACCEL)
    return angle_override

  def apply_override_angle_progressive(self, lat_active: bool, driverTorque: float, vEgo: float,
                                    VM: VehicleModel, unwind_scale: float = 1.0) -> float:
    """
    Converts steering torque to steering rotation rate.
    Physically angle rate is related to viscous damping of tires rotating on the ground.
    Here, however, the angle rate target is obtained from lateral jerk limit
    as a reasonable safe rate which decays quadratically with vehicle speed.
    """
    if not lat_active:
      self.override_angle_accu = 0
      return 0

    # unwind accumulator toward zero if the previous loop saturated (apply_steer_angle_limits_vm)
    unwind = (self.coop_apply_angle_last - self.coop_apply_angle_last_sat) * unwind_scale
    if self.override_angle_accu * unwind > 0:
      unwind = apply_bounds(unwind, abs(self.override_angle_accu))
      self.override_angle_accu -= unwind

    # torque biasing emulates the steering centering when released:
    if self.override_angle_accu > 0 and abs(vEgo) > 0.1:
      torque_biased = driverTorque - STEER_OVERRIDE_MIN_TORQUE
    elif self.override_angle_accu < 0 and abs(vEgo) > 0.1:
      torque_biased = driverTorque + STEER_OVERRIDE_MIN_TORQUE
    else:
      # when override_angle_accu is reset this turns off  everything
      torque_biased = apply_deadzone(driverTorque, STEER_OVERRIDE_MIN_TORQUE)

    # higher rate when centering
    angle_override_delta = calc_override_angle_delta_limited(torque_biased, vEgo, VM,
                          STEER_OVERRIDE_MAX_LAT_JERK if (torque_biased * self.override_angle_accu) > 0
                          else STEER_OVERRIDE_MAX_LAT_JERK_CENTERING)

    # ramp the angle
    new_override_angle_accu = self.override_angle_accu + angle_override_delta
    # snap to 0 if sign changes and driver torque is steering centering zone
    if (new_override_angle_accu * self.override_angle_accu) < 0 and abs(driverTorque) < STEER_OVERRIDE_MIN_TORQUE:
      new_override_angle_accu = 0

    self.override_angle_accu = new_override_angle_accu

    return self.override_angle_accu

  def apply_override_angle_combined(self, lat_active: bool, driverTorque: float, vEgo: float, VM: VehicleModel) -> float:
    """
    Vehicle-speed based transition between direct and progressive override control modes.
    Fuzzes the two modes based on the capabilities of the direct control mode.
    """
    if not lat_active:
      return 0

    # calculate capability of direct angle override (fully active above ~36kph)
    direct_override_capability = (calc_override_angle_limited(STEER_OVERRIDE_TORQUE_RANGE, vEgo, VM, STEER_OVERRIDE_MAX_LAT_ACCEL) /
                   get_steer_from_lat_accel(STEER_OVERRIDE_MAX_LAT_ACCEL, vEgo, VM))

    # Direct override capability approaches 0 at standstill as desired lat accel approaches infinity.
    # Use that as a scale to remove direct override influence at low speeds, ensuring a single
    # gain contribution in the torque-to-angle conversion.
    direct_control = direct_override_capability
    progressive_control = 1.0 - direct_control

    angle_override_direct = self.apply_override_angle_direct(lat_active, driverTorque, vEgo, VM)
    angle_override_progressive = self.apply_override_angle_progressive(lat_active, driverTorque, vEgo, VM,
                                                                 unwind_scale=progressive_control)

    return angle_override_direct * direct_control + angle_override_progressive * progressive_control

  def resume_steer_desired_rate_limit(self, lat_active: bool, apply_angle: float, steering_angle: float) -> float:
    """Limits steering wheel acceleration when resuming steering"""
    if not lat_active:
      # reset and bypass
      self.resume_rate_limiter_delta.reset(0)
      self.resume_rate_limiter.reset(steering_angle)
      return steering_angle

    angle_rate_delta_lim = self.resume_rate_limiter_delta.update(CarControllerParams.ANGLE_LIMITS.MAX_ANGLE_RATE,
                                                         STEER_RESUME_RATE_LIMIT_RAMP_RATE * DT_LAT_CTRL**2)
    apply_angle_lim = self.resume_rate_limiter.update(apply_angle, angle_rate_delta_lim)
    return apply_angle_lim

  def update(self, apply_angle, lat_active, CP_SP: structs.CarParamsSP, CS: structs.CarState, VM: VehicleModel) -> CoopSteeringDataSP:
    # estimate real steering angle by adding rate to the tesla filtered angle
    steeringAngleDegPhaseLead = CS.out.steeringAngleDeg + CS.out.steeringRateDeg / STEERING_DEG_PHASE_LEAD_COEFF

    angle_coop_enabled = CP_SP.flags & TeslaFlagsSP.COOP_STEERING.value

    # avoid sudden rotation on engagement
    apply_angle = self.resume_steer_desired_rate_limit(lat_active, apply_angle, steeringAngleDegPhaseLead)

    if angle_coop_enabled:
      self.debug_angle_desired_limited = apply_angle #! debug

      apply_angle += self.apply_override_angle_combined(lat_active, CS.out.steeringTorque, CS.out.vEgo, VM)

    # final rate limit - matching panda safety
    self.coop_apply_angle_last = apply_angle
    self.coop_apply_angle_last_sat = apply_steer_angle_limits_vm(apply_angle, self.coop_apply_angle_last_sat, CS.out.vEgoRaw,
                                                    CS.out.steeringAngleDeg, lat_active, CoopSteeringCarControllerParams, VM)

    return CoopSteeringDataSP(self.coop_apply_angle_last_sat, lat_active)
