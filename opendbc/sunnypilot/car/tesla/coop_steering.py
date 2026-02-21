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

# limit steering acceleration when engaging
STEER_RESUME_RATE_LIMIT_RAMP_RATE = 300 # deg/s^2

class CoopSteeringCarControllerParams(CarControllerParams):
  ANGLE_LIMITS = replace(CarControllerParams.ANGLE_LIMITS, MAX_ANGLE_RATE=5)

# angle override # todo implement steering torque inertia compensation to increase gains
STEER_OVERRIDE_MIN_TORQUE = 0.5 # Nm - based on typical steering bias + noise - used for the deadzone
STEER_OVERRIDE_MAX_TORQUE = 2.5 # Nm - typical torque before EPS disengages due to hands_on_level=3
STEER_OVERRIDE_TORQUE_RANGE = STEER_OVERRIDE_MAX_TORQUE - STEER_OVERRIDE_MIN_TORQUE

STEER_OVERRIDE_MAX_LAT_ACCEL = 2.0 # m/s^2 - determines angle rate - speed dependent - similar to Tesla comfort steering mode
STEER_OVERRIDE_TARGET_ANGLE_MAX = CarControllerParams.ANGLE_LIMITS.STEER_ANGLE_MAX  # deg

# override angle ramp control
STEER_OVERRIDE_DELTA_GAIN_LIMIT = 125 # deg/s/Nm
STEER_OVERRIDE_DELTA_GAIN_LIMIT_CENTERING = CoopSteeringCarControllerParams.ANGLE_LIMITS.MAX_ANGLE_RATE / DT_LAT_CTRL / STEER_OVERRIDE_TORQUE_RANGE
STEER_OVERRIDE_OPPOSING_DELTA_CONSUME_GAIN = 1

# model fighting mitigation
STEER_DESIRED_LIMITER_ALLOW_SPEED = 6 # m/s - below this speed the desired angle limiter is active
STEER_DESIRED_LIMITER_ACCEL = 100 # deg/s^2 when override angle ramp is active
STEER_DESIRED_LIMITER_OVERRIDE_ACTIVE_COUNTER = 3.0 # second


CoopSteeringDataSP = namedtuple("CoopSteeringDataSP",
                                ["steeringAngleDeg", "lat_active"])

def get_steer_from_lat_accel(lat_accel, vEgo: float, VM: VehicleModel):
  """Calculate the maximum steering angle based on lateral acceleration."""
  curvature = lat_accel / (max(1, vEgo) ** 2)  # 1/m
  return math.degrees(VM.get_steer_from_curvature(curvature, vEgo, 0))  # deg


def apply_bounds(signal: float, limit: float) -> float:
  """Limit input to a range."""
  return float(np.clip(signal, -limit, limit))


def apply_deadzone(signal: float, deadzone: float) -> float:
  """Apply deadzone to input."""
  return signal - apply_bounds(signal, deadzone)


def calc_override_angle_limited(torque: float, vEgo: float, VM: VehicleModel, lat_accel) -> float:
  """
  Map driver torque to lateral acceleration and convert to steering angle.
  """

  return torque * get_override_torque_to_angle(vEgo, VM, lat_accel)


def get_override_torque_to_angle(vEgo: float, VM: VehicleModel, lat_accel: float) -> float:
  """
  Convert effective override torque to steering angle gain.
  """

  # lateral accel is linear in respect to angle so it's fine to interpolate it with torque
  steer_from_lat_accel = apply_bounds(get_steer_from_lat_accel(lat_accel, vEgo, VM), STEER_OVERRIDE_TARGET_ANGLE_MAX)
  return steer_from_lat_accel / STEER_OVERRIDE_TORQUE_RANGE


def calc_override_angle_delta_limit(torque: float, gain_limit: float) -> float:
  """
  Convert torque magnitude to a per-step steering angle delta limit.
  """
  delta_gain_limit_max = CoopSteeringCarControllerParams.ANGLE_LIMITS.MAX_ANGLE_RATE / DT_LAT_CTRL / STEER_OVERRIDE_TORQUE_RANGE
  return torque * min(gain_limit, delta_gain_limit_max) * DT_LAT_CTRL


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


class SteerJerkLimiter:
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
    self.apply_angle_last = 0
    self.coop_apply_angle_sat_last = 0
    self.angle_override = 0
    self.override_active_counter = 0  # Counter for how many cycles torque is below threshold
    self.resume_rate_limiter_delta = SteerRateLimiter()
    self.resume_rate_limiter = SteerRateLimiter()
    self.override_accel_rate_limiter = SteerJerkLimiter()
    self.debug_angle_desired_limited = 0

  def reset_override_state(self, apply_angle: float) -> None:
    self.apply_angle_last = apply_angle
    self.angle_override = 0
    self.coop_apply_angle_sat_last = apply_angle
    self.override_accel_rate_limiter.reset(apply_angle)

  def update_override_angle(self, apply_angle_delta: float,
                                         driver_torque: float, vEgo: float, VM: VehicleModel) -> float:
    """
    Update angle_override toward the driver torque target subject to torque-based rate limits.
    """
    # Target angle
    driver_torque_with_deadzone = apply_deadzone(driver_torque, STEER_OVERRIDE_MIN_TORQUE)
    torque_to_angle = get_override_torque_to_angle(vEgo, VM, STEER_OVERRIDE_MAX_LAT_ACCEL)
    angle_override_target = driver_torque_with_deadzone * torque_to_angle
    target_error = angle_override_target - self.angle_override

    # Holding torque for centering and driving steering override rate determination
    if abs(vEgo) > 0.1:
      holding_torque = self.angle_override / torque_to_angle
    else:
      holding_torque = 0

    hold_torque_delta = driver_torque_with_deadzone - holding_torque

    delta_limit_away = calc_override_angle_delta_limit(abs(hold_torque_delta), STEER_OVERRIDE_DELTA_GAIN_LIMIT)
    delta_limit_center = calc_override_angle_delta_limit(abs(hold_torque_delta), STEER_OVERRIDE_DELTA_GAIN_LIMIT_CENTERING)

    down_step = delta_limit_center if self.angle_override > 0 else delta_limit_away
    up_step = delta_limit_center if self.angle_override < 0 else delta_limit_away

    angle_override_delta = float(np.clip(target_error, -down_step, up_step))

    # subtract same-direction angle delta already applied upstream
    if angle_override_delta * apply_angle_delta > 0:
      angle_override_delta = angle_override_delta - apply_bounds(apply_angle_delta, abs(angle_override_delta))
    elif angle_override_delta * apply_angle_delta < 0:
      opposing_consume_ratio = STEER_OVERRIDE_OPPOSING_DELTA_CONSUME_GAIN * max(0.0, abs(hold_torque_delta) / STEER_OVERRIDE_TORQUE_RANGE)
      angle_override_delta = angle_override_delta - opposing_consume_ratio * apply_angle_delta

    # ramp the angle
    self.angle_override += angle_override_delta

    return self.angle_override

  def unwind_override_angle_progressive(self, sat_error: float) -> None:
    """Apply same-frame anti-windup after the final steering angle limiter."""
    if self.angle_override * sat_error > 0:
      sat_error = apply_bounds(sat_error, abs(self.angle_override))
      self.angle_override -= sat_error

  def overriding_steer_desired_accel_limit(self, apply_angle: float, vEgo: float, steeringTorque: float) -> float:
    """
    Acceleration rate limiter - limits acceleration but allows for quick deceleration (no overshoot)
    """

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

  def resume_steer_desired_rate_limit(self, lat_active: bool, apply_angle: float, steering_rate_deg: float) -> float:
    """Limits steering wheel acceleration when resuming steering"""
    if not lat_active:
      # reset and bypass
      self.resume_rate_limiter_delta.reset(0)
      self.resume_rate_limiter.reset(apply_angle + steering_rate_deg * DT_LAT_CTRL)
      return apply_angle

    angle_rate_delta_lim = self.resume_rate_limiter_delta.update(CarControllerParams.ANGLE_LIMITS.MAX_ANGLE_RATE,
                                                         STEER_RESUME_RATE_LIMIT_RAMP_RATE * DT_LAT_CTRL**2)
    apply_angle_lim = self.resume_rate_limiter.update(apply_angle, angle_rate_delta_lim)
    return apply_angle_lim

  def update(self, apply_angle, lat_active, CP_SP: structs.CarParamsSP, CS: structs.CarState, VM: VehicleModel) -> CoopSteeringDataSP:
    angle_coop_enabled = CP_SP.flags & TeslaFlagsSP.COOP_STEERING.value

    # avoid sudden rotation on engagement
    apply_angle = self.resume_steer_desired_rate_limit(lat_active, apply_angle, CS.out.steeringRateDeg)

    if not lat_active or not angle_coop_enabled:
      self.reset_override_state(apply_angle)
      return CoopSteeringDataSP(apply_angle, lat_active)

    # apply_angle = self.overriding_steer_desired_accel_limit(apply_angle, CS.out.vEgo, CS.out.steeringTorque)
    self.debug_angle_desired_limited = apply_angle #! debug

    apply_angle_delta = apply_angle - self.apply_angle_last
    self.apply_angle_last = apply_angle
    apply_angle += self.update_override_angle(apply_angle_delta, CS.out.steeringTorque, CS.out.vEgo, VM)

    # final rate limit - matching panda safety
    self.coop_apply_angle_sat_last = apply_steer_angle_limits_vm(apply_angle, self.coop_apply_angle_sat_last, CS.out.vEgoRaw,
                                                    CS.out.steeringAngleDeg, lat_active, CoopSteeringCarControllerParams, VM)
    sat_error = apply_angle - self.coop_apply_angle_sat_last
    self.unwind_override_angle_progressive(sat_error)

    return CoopSteeringDataSP(self.coop_apply_angle_sat_last, lat_active)
