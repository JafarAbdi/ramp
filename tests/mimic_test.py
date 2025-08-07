"""Test the mimic joints class."""

import pathlib

import numpy as np
import pytest

from ramp import load_robot_model, RobotState

FILE_PATH = pathlib.Path(__file__).parent


def test_invalid_joint():
    """Test that an error is raised when a joint is not found."""
    with pytest.raises(
        ValueError,
        match="Joint 'panda_finger_joint2' is a mimic joint, which is not allowed in groups.",
    ):
        robot_model = load_robot_model(FILE_PATH / "group_joint_mimic.toml")


def test_mimic_joint():
    """Test the mimic joints of a robot."""
    robot_model = load_robot_model(
        FILE_PATH / ".." / "robots" / "panda" / "configs.toml",
    )
    group_name = "gripper"
    robot_state = RobotState(robot_model)
    number_of_mimic_joints = 1
    assert (
        len(robot_model.mimic_joint_indices) == 1
    ), f"Expected 1 mimic joints, found {len(robot_model.mimic_joint_indices)}"
    robot_state.set_group_qpos(group_name, [0.0])
    assert np.allclose(
        robot_state.qpos[robot_model.mimic_joint_indices],
        np.zeros(number_of_mimic_joints),
    )
    robot_state.set_group_qpos(group_name, [0.5])
    assert np.allclose(
        robot_state.qpos[robot_model.mimic_joint_indices],
        np.array([0.5]),
        atol=1e-6,
    )
