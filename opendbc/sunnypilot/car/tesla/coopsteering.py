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

STEER_OVERRIDE_MIN_TORQUE = 0.5 # Nm - based on typical steering bias + noise
STEER_OVERRIDE_MAX_TORQUE = 2.5 # Nm max torque before EPS disengages, LKAS takes over at 1.8Nm
STEER_OVERRIDE_MAX_LAT_ACCEL = 2.0 # m/s^2 - similar to Tesla comfort steering mode
STEER_OVERRIDE_GAIN_LIMIT = 10 # jerky but stable


def get_steer_from_lat_accel(lat_accel, v_ego_raw: float, VM: VehicleModel):
  """Calculate the maximum steering angle based on lateral acceleration."""
  max_curvature = lat_accel / (max(1, v_ego_raw) ** 2)  # 1/m
  return math.degrees(VM.get_steer_from_curvature(max_curvature, v_ego_raw, 0))  # deg

def applyOverrideAngle(apply_angle: float, driverTorque: float, vEgo: float, VM: VehicleModel) -> float:
  # ignore torque offset and disturbances
  steering_torque_with_deadzone = driverTorque - np.clip(driverTorque, -STEER_OVERRIDE_MIN_TORQUE, STEER_OVERRIDE_MIN_TORQUE)
  override_ratio_signed = steering_torque_with_deadzone / (STEER_OVERRIDE_MAX_TORQUE - STEER_OVERRIDE_MIN_TORQUE)
  # emulate power steering resistance - lateral acceleration maps to steering torque via steering rack
  override_angle_target = get_steer_from_lat_accel(STEER_OVERRIDE_MAX_LAT_ACCEL * override_ratio_signed, vEgo, VM) # todo this assumes we are turning from straight, add actual lat acceleration

  # to prevent jerky steering limit the override angle to this linear torque to angle mapping
  # todo replace with some dynamic trajectory function
  override_angle_gain_limit = STEER_OVERRIDE_GAIN_LIMIT * steering_torque_with_deadzone
  override_angle_target = np.clip(override_angle_target, -override_angle_gain_limit, override_angle_gain_limit)

  return apply_angle + override_angle_target


CoopSteeringDataSP = namedtuple("CoopSteeringDataSP",
                                ["control_type", "steeringAngleDeg"])

class CoopSteeringCarController:
  def __init__(self):
    super().__init__()
    self.coop_steering = CoopSteeringDataSP(False, 0)

  @staticmethod
  def coop_steering_update(CC: structs.CarControl, CC_SP: structs.CarControlSP, CS: structs.CarState, VM: VehicleModel) -> CoopSteeringDataSP:
    coop_steering = get_param(CC_SP.params, "TeslaCoopSteering", "0") == "True"
    control_type = 2 if coop_steering else 1

    steering_angle_with_override = applyOverrideAngle(CC.actuators.steeringAngleDeg, CS.out.steeringTorque, CS.out.vEgoRaw, VM)

    return CoopSteeringDataSP(control_type, steering_angle_with_override)

  def update(self, CC: structs.CarControl, CC_SP: structs.CarControlSP, CS: structs.CarState) -> None:
    self.coop_steering = self.coop_steering_update(CC, CC_SP, CS, self.VM)


