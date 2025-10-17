# ruff: noqa
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
from enum import Enum, auto
from dataclasses import dataclass

from opendbc.car import structs
from opendbc.car.vehicle_model import VehicleModel
from opendbc.car import DT_CTRL

# Steer pause logic
STEER_PAUSE_MIN_TORQUE = 0.6
STEER_PAUSE_ALLOW_SPEED = 7.0  # enabling for higher speed can be dangerous if accidentally triggered - tied to LKAS_OVERRIDE_ON_SPEED
STEER_PAUSE_WAIT_TIME = 0.5  # s - wait time before disengaging after engagement with a stalk
STEER_PAUSE_HOLD_TARGET_DEVIATION = 10  # deg - max allowed deviation from target angle when in hold state


def est_holding_torque(steering_angle: float, vEgo: float, VM: VehicleModel) -> float:
    """Estimate torque necessary to hold steering wheel in place.
    
    Args:
        steering_angle: Current steering angle in degrees
        vEgo: Vehicle speed in m/s
        VM: VehicleModel instance
        
    Returns:
        Estimated holding torque in Nm
    """
    lat_accel = get_lat_accel_from_steer(steering_angle, vEgo, VM)
    return lat_accel * 0.5  # Simple model: torque proportional to lateral acceleration


def override_above_holding_torque(driver_torque: float, holding_torque: float) -> bool:
    """
    Determines whether override torque is enough to hold the steering wheel in place (outward)
    or if input is above min torque threshold (inward).
    
    Args:
        driver_torque: Torque applied by the driver in Nm
        holding_torque: Estimated torque needed to hold current steering angle
        
    Returns:
        True if driver torque exceeds holding torque or minimum threshold
    """
    if holding_torque > 0:  # same sign as steering angle
        torque_override_left = -STEER_PAUSE_MIN_TORQUE
        torque_override_right = max(holding_torque, STEER_PAUSE_MIN_TORQUE)
    else:
        torque_override_left = min(holding_torque, -STEER_PAUSE_MIN_TORQUE)
        torque_override_right = STEER_PAUSE_MIN_TORQUE

    return not (torque_override_left <= driver_torque <= torque_override_right)


def get_lat_accel_from_steer(steer: float, v_ego: float, VM: VehicleModel):
  """Calculate the lateral acceleration based on steering angle."""
  curvature = VM.calc_curvature(math.radians(steer), v_ego, 0)  # 1/m
  return curvature * v_ego ** 2  # m/s^2


class LateralPauseState(Enum):
    INIT_WAIT = auto()
    NORMAL = auto()
    PAUSE = auto()
    REENGAGE_HOLD_WAIT = auto()

    def __str__(self):
        return self.name


@dataclass
class PauseStateManager:
    """Manages the state machine for lateral control pause functionality.
    
    Handles transitions between different pause states and tracks time spent in each state.
    """
    state: LateralPauseState = LateralPauseState.INIT_WAIT
    state_time: float = 0.0

    def reset_time(self) -> None:
        """Reset the time spent in the current state to zero."""
        self.state_time = 0.0

    def update_state(self, new_state: LateralPauseState) -> None:
        """Update the state and reset the timer.
        
        Args:
            new_state: The new state to transition to
        """
        if self.state != new_state:
            self.state = new_state
            self.reset_time()

    def tick(self, dt: float) -> None:
        """Update the time spent in the current state.
        
        Args:
            dt: Time elapsed since last update in seconds
        """
        self.state_time += dt

    def time_in_state(self) -> float:
        """Get the time spent in the current state.
        
        Returns:
            Time in seconds spent in current state
        """
        return self.state_time


class PauseManager:
  def __init__(self):
    self.psm = PauseStateManager()

  def reset_pause_state(self):
    self.psm.update_state(LateralPauseState.INIT_WAIT)
    self.psm.reset_time()

  def update_pause_state(self, angle_planned: float, CS: structs.CarState, VM: VehicleModel):
    torque_hold = est_holding_torque(CS.out.steeringAngleDeg, CS.out.vEgoRaw, VM)
    torque_override = override_above_holding_torque(CS.out.steeringTorque, torque_hold)

    # Engage conditions (when to enter or stay in LATERAL_PAUSED)
    should_disengage = (
      CS.out.vEgoRaw < STEER_PAUSE_ALLOW_SPEED # todo add hysteresis
      and self.psm.time_in_state() > STEER_PAUSE_WAIT_TIME
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
    is_far_from_target = abs(CS.out.steeringAngleDeg - angle_planned) > STEER_PAUSE_HOLD_TARGET_DEVIATION
    should_hold_wait = (
      CS.out.steeringRateDeg == 0
      and CS.hands_on_level < 3
      and not is_far_from_target
      # and not CS.out.standstill
    )
    should_reengage_from_hold = (self.psm.time_in_state() > 1.0 or should_reengage_released) \
      and not should_disengage

    # State transitions
    if self.psm.state == LateralPauseState.INIT_WAIT and self.psm.time_in_state() > STEER_PAUSE_WAIT_TIME:
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

    lat_pause = self.psm.state == LateralPauseState.PAUSE \
        or self.psm.state == LateralPauseState.REENGAGE_HOLD_WAIT
    return lat_pause

