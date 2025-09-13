"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
import numpy as np
from collections import namedtuple

from opendbc.car import structs
from opendbc.car.vehicle_model import VehicleModel
from opendbc.sunnypilot.car import get_param
from openpilot.common.params import Params

LKAS_OVERRIDE_OFF_SPEED = 6.0 # LKAS coop steering completly off below
LKAS_OVERRIDE_ON_SPEED = 7.0 # LKAS coop steering completly on above
LKAS_OVERRIDE_ON_TORQUE = 2.0 # LKAS coop usually On above this torque
LKAS_OVERRIDE_OFF_TORQUE = 1.3 # LKAS coop usually Off below this torque

STEER_OVERRIDE_MIN_TORQUE = 0.5 # Nm - based on typical steering bias + noise
STEER_OVERRIDE_MAX_TORQUE = 2.5 # Nm max torque before EPS disengages, LKAS takes over at 1.8Nm
STEER_OVERRIDE_MAX_LAT_ACCEL = 2.0 # m/s^2 - similar to Tesla comfort steering mode 
STEER_OVERRIDE_GAIN_LIMIT = 6 # stability and smoothness in angle control mode or LKAS low speed

ALLOW_STEER_PAUSE_SPEED = LKAS_OVERRIDE_ON_SPEED

def get_steer_from_lat_accel(lat_accel, v_ego: float, VM: VehicleModel):
  """Calculate the maximum steering angle based on lateral acceleration."""
  max_curvature = lat_accel / (max(1, v_ego) ** 2)  # 1/m
  return math.degrees(VM.get_steer_from_curvature(max_curvature, v_ego, 0))  # deg

def calc_override_angle(apply_angle: float, driverTorque: float, vEgo: float, VM: VehicleModel) -> float:
  """Convert driver torque to lateral acceleration and apply override angle."""
  # ignore torque sensor offset and disturbances
  steering_torque_with_deadzone = driverTorque - np.clip(driverTorque, -STEER_OVERRIDE_MIN_TORQUE, STEER_OVERRIDE_MIN_TORQUE)
  max_override_torque = (STEER_OVERRIDE_MAX_TORQUE - STEER_OVERRIDE_MIN_TORQUE)

  # lateral acc is linear in respect to angle so it's fine to interpolate it with torque
  torque_to_angle = get_steer_from_lat_accel(STEER_OVERRIDE_MAX_LAT_ACCEL, vEgo, VM) / max_override_torque
  # limit the gain to prevent jerkiness and instability
  override_angle_target = steering_torque_with_deadzone * min(torque_to_angle, STEER_OVERRIDE_GAIN_LIMIT)

  return apply_angle + override_angle_target

def lkas_compensation(apply_angle: float, apply_angle_last: float, steering_angle: float, driverTorque: float, vEgo: float) -> float:
  # lkas contribution is done by the car and is a difference betwen out command and measured angle
  lkas_angle = steering_angle - apply_angle_last

  # smooth transition to LKAS based on enable torque
  lkas_angle = np.interp(abs(driverTorque),
                         [LKAS_OVERRIDE_OFF_TORQUE, LKAS_OVERRIDE_ON_TORQUE],
                         [0, lkas_angle])

  # get out of the way if below  speed for LKAS coop steering
  if vEgo < LKAS_OVERRIDE_OFF_SPEED:
    lkas_angle = 0

  return apply_angle - lkas_angle


CoopSteeringDataSP = namedtuple("CoopSteeringDataSP",
                                ["control_type", "lat_pause", "steeringAngleDeg"])

class CoopSteeringCarState:
  def __init__(self):
    self.enabled = False

  def controls_disengage_cond(self, ret: structs.CarState) -> bool:
    self.enabled = Params().get_bool("TeslaCoopSteering")
    
    if self.enabled and ret.vEgo < ALLOW_STEER_PAUSE_SPEED:
      # ignore hands on level when cooperative steering is enabled
      return ret.steeringDisengage and not self.hands_on_level >= 3
    return ret.steeringDisengage

class CoopSteeringCarController:
  def __init__(self):
    super().__init__()
    self.enabled = False
    self.coop_steering = CoopSteeringDataSP(False, False, 0)

  def steer_pause_state(self, CC: structs.CarControl, CS: structs.CarState) -> bool:
    if self.coop_steering.lat_pause:
      # keep disengaged while:
      lat_pause_req = (
          CS.hands_on_level > 0
          or CS.out.standstill
          or CS.out.steeringRateDeg != 0
      )
    else:
      lat_pause_req = CS.hands_on_level == 3 # todo lower threshold for low speed / low angle
    return lat_pause_req

  def coop_steering_update(self, CC: structs.CarControl, CC_SP: structs.CarControlSP, CS: structs.CarState, VM: VehicleModel) -> CoopSteeringDataSP:
    self.enabled = get_param(CC_SP.params, "TeslaCoopSteering", "0") == "True"
    control_type = 2 if self.enabled else 1

    lat_pause = False
    if self.enabled and CC.latActive:
      lat_pause = self.steer_pause_state(CC, CS)

    apply_angle_with_override = calc_override_angle(CC.actuators.steeringAngleDeg, CS.out.steeringTorque, CS.out.vEgoRaw, VM)
    if control_type == 2: # LKAS
      apply_angle_with_override = lkas_compensation(apply_angle_with_override, self.apply_angle_last, CS.out.steeringAngleDeg,
                                                    CS.out.steeringTorque, CS.out.vEgoRaw)

    return CoopSteeringDataSP(control_type, lat_pause, apply_angle_with_override)

  def update(self, CC: structs.CarControl, CC_SP: structs.CarControlSP, CS: structs.CarState) -> CoopSteeringDataSP:
    self.coop_steering = self.coop_steering_update(CC, CC_SP, CS, self.VM)
    return self.coop_steering

