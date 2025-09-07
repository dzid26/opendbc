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

ButtonType = structs.CarState.ButtonEvent.Type


class CarStateExt:
  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParamsSP):
    self.CP = CP
    self.CP_SP = CP_SP
    self.high_beam_state = 0

  def update(self, ret: structs.CarState, can_parsers: dict[StrEnum, CANParser]) -> None:
    cp_party = can_parsers[Bus.party]
    prev_high_beam_state = self.high_beam_state
    # DBC: UI_warning.highBeam (bool)
    self.high_beam_state = int(cp_party.vl["UI_warning"].get("highBeam", 0))
    if not getattr(ret, 'buttonEvents', None):
      ret.buttonEvents = []
    ret.buttonEvents = [*ret.buttonEvents, *create_button_events(self.high_beam_state, prev_high_beam_state, {1: ButtonType.lkas})]
