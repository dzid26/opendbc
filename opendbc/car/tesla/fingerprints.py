""" AUTO-FORMATTED USING opendbc/car/debug/format_fingerprints.py, EDIT STRUCTURE THERE."""
# Pre-3-bit-DAS_steeringControlType fingerprints removed: those EPS firmwares pack
# DAS_steeringControlType as a 2-bit field at bits 22-23, incompatible with the
# 3-bit signal definition (bits 21-23) used since Tesla's signal change.
# Reverts the FSD-14 detection mechanism from commit 58f03320267c940022d7c53a6a303aebfe3b38ef.
from opendbc.car.structs import CarParams
from opendbc.car.tesla.values import CAR
from opendbc.sunnypilot.car.fingerprints_ext import merge_fw_versions
from opendbc.sunnypilot.car.tesla.fingerprints_ext import FW_VERSIONS_EXT

Ecu = CarParams.Ecu

FW_VERSIONS = {
  CAR.TESLA_MODEL_3: {
    (Ecu.eps, 0x730, None): [
      b'TeM3_E014p10_0.0.0 (24),E014.20.2',
      b'TeMYG4_Main_0.0.0 (77),E4H015.04.5',
      b'TeMYG4_Main_0.0.0 (77),E4HP015.04.5',
      b'TeMYG4_Main_0.0.0 (78),E4H015.05.0',
      b'TeMYG4_Main_0.0.0 (78),E4HP015.05.0',
    ],
  },
  CAR.TESLA_MODEL_Y: {
    (Ecu.eps, 0x730, None): [
      b'TeM3_E014p10_0.0.0 (24),YP002.21.2',
      b'TeMYG4_Legacy3Y_0.0.0 (6),Y4003.04.0',
      b'TeMYG4_Main_0.0.0 (77),Y4003.05.4',
      b'TeMYG4_Main_0.0.0 (78),Y4003.06.0',
    ],
  },
}

FW_VERSIONS = merge_fw_versions(FW_VERSIONS, FW_VERSIONS_EXT)
