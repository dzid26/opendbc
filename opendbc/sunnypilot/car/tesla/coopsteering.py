"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from collections import namedtuple

from opendbc.car import structs
from opendbc.sunnypilot.car import get_param

CoopSteeringDataSP = namedtuple("CoopSteeringDataSP",
                                ["control_type"])


class CoopSteeringCarController:
  def __init__(self):
    super().__init__()
    self.coop_steering = CoopSteeringDataSP(False)

  @staticmethod
  def coop_steering_status_update(CC: structs.CarControl, CC_SP: structs.CarControlSP) -> CoopSteeringDataSP:
    coop_steering = get_param(CC_SP.params, "TeslaCoopSteering", False)
    control_type = 2 if coop_steering else 1

    return CoopSteeringDataSP(control_type)

  def update(self, CC: structs.CarControl, CC_SP: structs.CarControlSP) -> None:
    self.coop_steering = self.coop_steering_status_update(CC, CC_SP)
