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
STEER_OVERRIDE_MIN_TORQUE = 0.5 # Nm - based on typical steering bias + noise
STEER_OVERRIDE_MAX_TORQUE = 2.5 # Nm max torque before EPS disengages

STEER_OVERRIDE_MAX_LAT_ACCEL = 2.0 # m/s^2 - determines angle rate - speed dependent - similar to Tesla comfort steering mode
STEER_OVERRIDE_TARGET_ANGLE_MAX = 2 * CarControllerParams.ANGLE_LIMITS.STEER_ANGLE_MAX  # deg

# angle ramping
STEER_OVERRIDE_CENTERING_MIN_SPEED = 0.1  # m/s, avoid springing back at standstill
STEER_OVERRIDE_TORQUE_RANGE = STEER_OVERRIDE_MAX_TORQUE - STEER_OVERRIDE_MIN_TORQUE
# stability and smoothness for angle ramp control at low speeds
STEER_OVERRIDE_DELTA_GAIN_LIMIT = 100 # deg/s/Nm
STEER_OVERRIDE_DELTA_GAIN_LIMIT_CENTERING = CoopSteeringCarControllerParams.ANGLE_LIMITS.MAX_ANGLE_RATE / DT_LAT_CTRL / STEER_OVERRIDE_TORQUE_RANGE
STEER_OVERRIDE_OPPOSING_DELTA_CONSUME_GAIN = 1

# model fighting mitigation
STEER_DESIRED_LIMITER_ALLOW_SPEED = 6 # m/s - below this speed the desired angle limiter is active
STEER_DESIRED_LIMITER_ACCEL = 100 # deg/s^2 when override angle ramp is active
STEER_DESIRED_LIMITER_OVERRIDE_ACTIVE_COUNTER = 3.0 # second


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
  """

  # lateral accel is linear in respect to angle so it's fine to interpolate it with torque
  torque_to_angle = get_steer_from_lat_accel(lat_accel, vEgo, VM) / STEER_OVERRIDE_TORQUE_RANGE
  return torque * min(torque_to_angle, STEER_OVERRIDE_TARGET_ANGLE_MAX / STEER_OVERRIDE_TORQUE_RANGE)


def calc_override_angle_delta_limit(torque: float, gain_limit: float) -> float:
  """
  Convert torque magnitude to a per-step steering angle delta limit.
  """
  gain_limit = min(gain_limit, CoopSteeringCarControllerParams.ANGLE_LIMITS.MAX_ANGLE_RATE / DT_LAT_CTRL / STEER_OVERRIDE_TORQUE_RANGE)
  return abs(torque) * gain_limit * DT_LAT_CTRL


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

  @property
  def override_angle_accu(self) -> float:
    return self.angle_override

  @override_angle_accu.setter
  def override_angle_accu(self, value: float) -> None:
    self.angle_override = value

  def reset_override_state(self, apply_angle: float) -> None:
    self.apply_angle_last = apply_angle
    self.angle_override = 0
    self.coop_apply_angle_sat_last = apply_angle
    self.override_accel_rate_limiter.reset(apply_angle)

  def get_angle_override_target(self, driverTorque: float, vEgo: float, VM: VehicleModel) -> float:
    """
    Convert driver torque to override target angle.
    """
    steering_torque_with_deadzone = apply_deadzone(driverTorque, STEER_OVERRIDE_MIN_TORQUE)
    return calc_override_angle_limited(steering_torque_with_deadzone, vEgo, VM, STEER_OVERRIDE_MAX_LAT_ACCEL)

  def update_override_angle_rate_limited(self, angle_override_target: float, apply_angle_delta: float,
                                         driverTorque: float, vEgo: float) -> float:
    """
    Update angle_override toward the driver torque target subject to torque-based rate limits.
    """
    torque_deadzone = apply_deadzone(driverTorque, STEER_OVERRIDE_MIN_TORQUE)

    # torque biasing emulates the steering centering when released:
    centering_active = self.angle_override != 0 and abs(vEgo) > STEER_OVERRIDE_CENTERING_MIN_SPEED
    if centering_active:
      torque_biased = driverTorque - np.sign(self.angle_override) * STEER_OVERRIDE_MIN_TORQUE
    else:
      torque_biased = torque_deadzone

    delta_limit_away = calc_override_angle_delta_limit(torque_deadzone, STEER_OVERRIDE_DELTA_GAIN_LIMIT)
    delta_limit_center = calc_override_angle_delta_limit(torque_biased, STEER_OVERRIDE_DELTA_GAIN_LIMIT_CENTERING)

    down_step = delta_limit_center if self.angle_override > 0 else delta_limit_away
    up_step = delta_limit_center if self.angle_override < 0 else delta_limit_away

    target_error = angle_override_target - self.angle_override
    angle_override_delta = float(np.clip(target_error, -down_step, up_step))

    # subtract same-direction angle delta already applied upstream
    if angle_override_delta * apply_angle_delta > 0:
      angle_override_delta = angle_override_delta - apply_bounds(apply_angle_delta, abs(angle_override_delta))
    elif angle_override_delta * apply_angle_delta < 0:
      opposing_consume_ratio = STEER_OVERRIDE_OPPOSING_DELTA_CONSUME_GAIN * max(0.0, abs(torque_biased) / STEER_OVERRIDE_TORQUE_RANGE)
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

  def resume_steer_desired_rate_limit(self, lat_active: bool, apply_angle: float) -> float:
    """Limits steering wheel acceleration when resuming steering"""
    if not lat_active:
      # reset and bypass
      self.resume_rate_limiter_delta.reset(0)
      self.resume_rate_limiter.reset(apply_angle)
      return apply_angle

    angle_rate_delta_lim = self.resume_rate_limiter_delta.update(CarControllerParams.ANGLE_LIMITS.MAX_ANGLE_RATE,
                                                         STEER_RESUME_RATE_LIMIT_RAMP_RATE * DT_LAT_CTRL**2)
    apply_angle_lim = self.resume_rate_limiter.update(apply_angle, angle_rate_delta_lim)
    return apply_angle_lim

  def update(self, apply_angle, lat_active, CP_SP: structs.CarParamsSP, CS: structs.CarState, VM: VehicleModel) -> CoopSteeringDataSP:
    angle_coop_enabled = CP_SP.flags & TeslaFlagsSP.COOP_STEERING.value

    # avoid sudden rotation on engagement
    apply_angle = self.resume_steer_desired_rate_limit(lat_active, apply_angle)

    if not lat_active or not angle_coop_enabled:
      self.reset_override_state(apply_angle)
      return CoopSteeringDataSP(apply_angle, lat_active)

    # apply_angle = self.overriding_steer_desired_accel_limit(apply_angle, CS.out.vEgo, CS.out.steeringTorque)
    self.debug_angle_desired_limited = apply_angle #! debug

    apply_angle_delta = apply_angle - self.apply_angle_last
    self.apply_angle_last = apply_angle
    angle_override_target = self.get_angle_override_target(CS.out.steeringTorque, CS.out.vEgo, VM)
    apply_angle += self.update_override_angle_rate_limited(angle_override_target, apply_angle_delta, CS.out.steeringTorque, CS.out.vEgo)

    # final rate limit - matching panda safety
    self.coop_apply_angle_sat_last = apply_steer_angle_limits_vm(apply_angle, self.coop_apply_angle_sat_last, CS.out.vEgoRaw,
                                                    CS.out.steeringAngleDeg, lat_active, CoopSteeringCarControllerParams, VM)
    sat_error = apply_angle - self.coop_apply_angle_sat_last
    self.unwind_override_angle_progressive(sat_error)

    return CoopSteeringDataSP(self.coop_apply_angle_sat_last, lat_active)
