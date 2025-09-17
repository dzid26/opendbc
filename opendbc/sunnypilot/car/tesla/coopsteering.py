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
from openpilot.common.params import Params

LKAS_OVERRIDE_OFF_SPEED = 6.0 # LKAS coop steering completly off below
LKAS_OVERRIDE_ON_SPEED = 6.7 # LKAS coop steering completly on above
LKAS_OVERRIDE_ON_TORQUE = 2.0 # LKAS coop usually On above this torque
LKAS_OVERRIDE_OFF_TORQUE = 1.3 # LKAS coop usually Off below this torque

STEER_OVERRIDE_MIN_TORQUE = 0.5 # Nm - based on typical steering bias + noise
STEER_OVERRIDE_MAX_TORQUE = 2.5 # Nm max torque before EPS disengages, LKAS takes over at 1.8Nm
STEER_OVERRIDE_MAX_LAT_ACCEL = 2.0 # m/s^2 - similar to Tesla comfort steering mode
STEER_OVERRIDE_GAIN_LIMIT = 8 # stability and smoothness in angle control mode or LKAS low speed
STEER_OVERRIDE_TORQUE_RANGE = STEER_OVERRIDE_MAX_TORQUE - STEER_OVERRIDE_MIN_TORQUE

STEER_PAUSE_ALLOW_SPEED = LKAS_OVERRIDE_ON_SPEED
STEER_PAUSE_WAIT_TIME = 0.5 # s - wait time before disengaging after engagement with a stalk

STEER_RESUME_RATE_LIMIT_RAMP_RATE = 100 # deg/s/10ms - controls rate of rise of angle rate limit, not angle directly

CoopSteeringDataSP = namedtuple("CoopSteeringDataSP",
                                ["control_type", "lat_pause", "steeringAngleDeg"])

class CoopSteeringCarState:
  def __init__(self):
    self.enabled = False

  def controls_disengage_cond(self, ret: structs.CarState) -> bool:
    self.enabled = Params().get_bool("TeslaCoopSteering")
    
    if self.enabled and ret.vEgo < STEER_PAUSE_ALLOW_SPEED:
      # ignore hands on level when cooperative steering is enabled
      return ret.steeringDisengage # todo fix this
    return ret.steeringDisengage


def get_steer_from_lat_accel(lat_accel, v_ego: float, VM: VehicleModel):
  """Calculate the maximum steering angle based on lateral acceleration."""
  curvature = lat_accel / (max(1, v_ego) ** 2)  # 1/m
  return math.degrees(VM.get_steer_from_curvature(curvature, v_ego, 0))  # deg

def calc_override_angle(apply_angle: float, driverTorque: float, vEgo: float, VM: VehicleModel) -> float:
  """Convert driver torque to lateral acceleration and apply override angle."""
  # ignore torque sensor offset and disturbances
  steering_torque_with_deadzone = driverTorque - np.clip(driverTorque, -STEER_OVERRIDE_MIN_TORQUE, STEER_OVERRIDE_MIN_TORQUE)
  
  # lateral acc is linear in respect to angle so it's fine to interpolate it with torque
  torque_to_angle = get_steer_from_lat_accel(STEER_OVERRIDE_MAX_LAT_ACCEL, vEgo, VM) / STEER_OVERRIDE_TORQUE_RANGE
  # limit the gain to prevent jerkiness and instability
  override_angle_target = steering_torque_with_deadzone * min(torque_to_angle, STEER_OVERRIDE_GAIN_LIMIT)

  return apply_angle + override_angle_target

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
  
  # get out of the way if below speed for LKAS coop steering
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

def determine_torque_override(driver_torque: float, holding_torque: float) -> bool:
  torque_override_outward = np.clip(holding_torque, -STEER_OVERRIDE_MAX_TORQUE, STEER_OVERRIDE_MAX_TORQUE)
  
  if holding_torque > 0: # same sign as CS.out.steeringAngleDeg
    torque_override_left = -STEER_OVERRIDE_MIN_TORQUE
    torque_override_right = max(torque_override_outward, STEER_OVERRIDE_MIN_TORQUE)
  else:
    torque_override_left = min(torque_override_outward, -STEER_OVERRIDE_MIN_TORQUE)
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

  def update_state(self, new_state: LateralPauseState):
    """ Update the state of the pause state machine and reset timer. """
    if self.state != new_state:
      print(f"DEBUG: (Pause state) {self.state} -> {new_state}")
      self.state = new_state
      self.state_time = 0.0
      
  def tick(self, dt: float):
    """ Update the time spent in the current state. """
    self.state_time += dt

class CoopSteeringCarController:
  def __init__(self):
    super().__init__()
    self.enabled = False
    self.coop_steering = CoopSteeringDataSP(False, False, 0)
    self.angle_rate_delta_lim = 0
    self.psm = PauseStateManager()


  def update_pause_state(self, CS: structs.CarState, VM: VehicleModel):
    torque_hold = est_holding_torque(CS.out.steeringAngleDeg, CS.out.vEgoRaw, VM)
    torque_override = determine_torque_override(CS.out.steeringTorque, torque_hold)
    
    # Engage conditions (when to enter or stay in LATERAL_PAUSED)
    should_disengage = (
      CS.out.vEgoRaw < STEER_PAUSE_ALLOW_SPEED
      and self.psm.state_time > STEER_PAUSE_WAIT_TIME
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
    should_hold_wait = (
      CS.out.steeringRateDeg == 0
      and CS.hands_on_level < 3
      # and not CS.out.standstill
    )
    should_reengage_from_hold = (self.psm.state_time > 1 or should_reengage_released) \
      and not should_disengage

    # State transitions
    if self.psm.state == LateralPauseState.INIT_WAIT and self.psm.state_time > STEER_PAUSE_WAIT_TIME:
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

  def resume_steer_rate_limit_ramp(self, resume, apply_angle: float, apply_angle_last: float) -> float:
    """Limits steering wheel acceleration when resuming steering after pause"""

    if resume:
      self.angle_rate_delta_lim = self.angle_rate_delta_lim + STEER_RESUME_RATE_LIMIT_RAMP_RATE * DT_CTRL
      apply_angle_lim = rate_limit(apply_angle, apply_angle_last, -self.angle_rate_delta_lim, self.angle_rate_delta_lim)
    else:
      # reset and bypass when paused
      self.angle_rate_delta_lim = 0
      apply_angle_lim = apply_angle

    return apply_angle_lim


  def coop_steering_update(self, CC: structs.CarControl, CC_SP: structs.CarControlSP, CS: structs.CarState, VM: VehicleModel) -> CoopSteeringDataSP:
    self.enabled = get_param(CC_SP.params, "TeslaCoopSteering", "0") == "True"

    if self.enabled and CC.latActive:
      control_type = 2  # LKAS mode todo: use CAN parser enums
      self.update_pause_state(CS, VM)
      lat_pause = self.psm.state != LateralPauseState.NORMAL
      apply_angle = calc_override_angle(CC.actuators.steeringAngleDeg, CS.out.steeringTorque, CS.out.vEgoRaw, VM)
      if control_type == 2: # LKAS
        apply_angle = lkas_compensation(apply_angle, self.coop_steering.steeringAngleDeg, CS.out.steeringAngleDeg,
                                                      CS.out.steeringTorque, CS.out.vEgoRaw)
    else:
      self.psm.update_state(LateralPauseState.INIT_WAIT)
      control_type = 1 # angle control mode
      lat_pause = False
      apply_angle = CC.actuators.steeringAngleDeg

    self.coop_steering = CoopSteeringDataSP(control_type, lat_pause, apply_angle)
    return self.coop_steering

  def update(self, CC: structs.CarControl, CC_SP: structs.CarControlSP, CS: structs.CarState) -> CoopSteeringDataSP:
    coop_steering = self.coop_steering_update(CC, CC_SP, CS, self.VM)

    # Replicate carcontroller behaviour here to apply steering wheel acceleration limit on all user hand overs
    lat_active = CC.latActive and CS.hands_on_level < 3 and not coop_steering.lat_pause
    apply_angle = self.resume_steer_acc_limit(lat_active, coop_steering.steeringAngleDeg, self.apply_angle_last)

    return CoopSteeringDataSP(coop_steering.control_type, coop_steering.lat_pause, apply_angle)
