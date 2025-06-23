"""Test the mimic joints class."""

import pathlib

import numpy as np
import pinocchio
import pytest

from ramp import load_robot_model, RobotState
from ramp.ik_solver import IKSolver
from ramp.exceptions import (
    MissingBaseLinkError,
    MissingAccelerationLimitError,
    MissingJointError,
    RobotDescriptionNotFoundError,
)

FILE_PATH = pathlib.Path(__file__).parent


def test_invalid_joint():
    """Test that an error is raised when a joint is not found."""
    with pytest.raises(
        ValueError,
        match="Joint 'right_outer_knuckle_joint' is a mimic joint, which is not allowed in groups.",
    ):
        robot_model = load_robot_model(FILE_PATH / "group_joint_mimic.toml")
