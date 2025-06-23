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
        match="Joint 'robotiq_85_left_inner_knuckle_joint' is a mimic joint, which is not allowed in groups.",
    ):
        robot_model = load_robot_model(FILE_PATH / "group_joint_mimic.toml")


ROBOTS = ["fr3_robotiq", "kinova", "ur5e"]


@pytest.mark.parametrize("robot_name", ROBOTS)
def test_mimic_joint(robot_name):
    """Test the mimic joints of a robot."""
    robot_model = load_robot_model(
        FILE_PATH / ".." / "robots" / robot_name / "configs.toml",
    )
    group_name = "gripper"
    robot_state = RobotState(robot_model)
    number_of_mimic_joints = 5
    assert (
        robot_model.mimic_joint_indices.size == 5
    ), f"Expected 5 mimic joints, found {robot_model.mimic_joint_indices.size}"
    robot_state.set_group_qpos(group_name, [0.0])
    assert np.allclose(
        robot_state.qpos[robot_model.mimic_joint_indices],
        np.zeros(number_of_mimic_joints),
    )
    robot_state.set_group_qpos(group_name, [0.5])
    assert np.allclose(
        robot_state.qpos[robot_model.mimic_joint_indices],
        np.array([-0.5, 0.5, -0.5, -0.5, 0.5]),
        atol=1e-6,
    )
