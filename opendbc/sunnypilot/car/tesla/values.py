"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from enum import IntFlag

class TeslaAngleMap:
  # angle request will be scaled by this map
  XP = [0, 90, 200, 330, 450]  # corresponds to model desired angle
  YP = [0, 90, 180, 270, 360]  # corresponds to steering command

class TeslaFlagsSP(IntFlag):
  HAS_VEHICLE_BUS = 1  # 3-finger infotainment press signal is present on the VEHICLE bus with the deprecated Tesla harness installed
  COOP_STEERING = 2 # virtual torque blending
  LKAS_STEERING = 2**2 # use LKAS steering interface to provide oem torque blending
  PAUSE_STEERING = 2**3 # pause steer on override at low speed


class TeslaSafetyFlagsSP:
  HAS_VEHICLE_BUS = 1
