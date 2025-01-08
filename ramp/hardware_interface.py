"""Hardware interface for reading and writing robot state."""

from abc import ABC, abstractmethod

import numpy as np

from ramp.robot_model import RobotModel
from ramp.robot_state import RobotState


class HardwareInterface(ABC):
    """Abstract class for hardware interface."""

    def __init__(self, robot_model: RobotModel):
        """Initialize hardware interface."""
        self.robot_model = robot_model

    @abstractmethod
    def read(self) -> RobotState:
        """Read robot state."""

    @abstractmethod
    def write(self, joint_names: list[str], ctrl: np.ndarray):
        """Write control commands to robot."""


class MockHardwareInterface(HardwareInterface):
    """Mock hardware interface for testing."""

    def __init__(self, robot_model: RobotModel, initial_robot_state: RobotState):
        """Initialize mock hardware interface.

        Args:
            robot_model: Robot model.
            initial_robot_state: Initial robot state.
        """
        super().__init__(robot_model)
        self._robot_state = initial_robot_state.clone()

    def read(self) -> RobotState:
        """Read robot state."""
        return self._robot_state.clone()

    def write(self, joint_names: list[str], ctrl: np.ndarray):
        """Write control commands to robot."""
        msg = "Mock hardware interface does not support writing."
        raise NotImplementedError(msg)
