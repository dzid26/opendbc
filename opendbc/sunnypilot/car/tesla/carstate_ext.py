"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from enum import StrEnum

from opendbc.car import Bus, create_button_events, structs
from opendbc.can.parser import CANParser
from opendbc.car.tesla.values import DBC, CANBUS
from opendbc.sunnypilot.car.tesla.values import TeslaFlagsSP
from opendbc.sunnypilot.car.tesla.coopsteering import CoopSteeringCarState

ButtonType = structs.CarState.ButtonEvent.Type


class CarStateExt(CoopSteeringCarState):
  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParamsSP):
    self.CP = CP
    self.CP_SP = CP_SP
    CoopSteeringCarState.__init__(self)

    self.infotainment_3_finger_press = 0

  def update(self, ret: structs.CarState, can_parsers: dict[StrEnum, CANParser]) -> None:
    # ret.steeringDisengage = self.controls_disengage_cond(ret)

    if self.CP_SP.flags & TeslaFlagsSP.HAS_VEHICLE_BUS:
      cp_adas = can_parsers[Bus.adas]

      prev_infotainment_3_finger_press = self.infotainment_3_finger_press
      self.infotainment_3_finger_press = int(cp_adas.vl["UI_status2"]["UI_activeTouchPoints"])

      ret.buttonEvents = [*create_button_events(self.infotainment_3_finger_press, prev_infotainment_3_finger_press,
                                                {3: ButtonType.lkas})]

  @staticmethod
  def get_parser(CP: structs.CarParams, CP_SP: structs.CarParamsSP) -> dict[StrEnum, CANParser]:
    messages = {}

    if CP_SP.flags & TeslaFlagsSP.HAS_VEHICLE_BUS:
      messages[Bus.adas] = CANParser(DBC[CP.carFingerprint][Bus.adas], [], CANBUS.vehicle)

    return messages
