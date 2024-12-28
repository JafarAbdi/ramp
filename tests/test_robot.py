"""Test the Robot class."""

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


def test_configs():
    # No mappings specified shouldn't raise an error
    robot_model = load_robot_model(
        FILE_PATH / ".." / "robots" / "planar_rrr" / "configs.toml"
    )


def test_rrr():
    """Test the Robot class with no gripper and tcp_link."""
    robot_model = load_robot_model(FILE_PATH / ".." / "robots" / "rrr" / "configs.toml")
    assert robot_model.base_link == "base_link"
    group_name = "arm"
    assert robot_model[group_name].tcp_link_name == "end_effector"
    assert robot_model[group_name].joints == ["joint1", "joint2", "joint3"]
    target_pose = [0.4, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0]
    # Using IK
    ik_solver = IKSolver(
        robot_model.model_filename,
        robot_model.base_link,
        robot_model[group_name].tcp_link_name,
    )
    initial_qpos = robot_model[group_name].named_states["home"]
    target_joint_positions = ik_solver.solve(target_pose, initial_qpos)
    assert target_joint_positions is not None

    robot_state = RobotState.from_named_state(robot_model, group_name, "home")
    # Using differential-ik
    assert robot_state.differential_ik(group_name, target_pose)
    pose = robot_state.get_frame_pose(robot_model[group_name].tcp_link_name)
    assert np.allclose(target_pose, pinocchio.SE3ToXYZQUAT(pose), atol=1e-3)

    # Should fail, rrr robot has end_effector_joint as revolute joint with [0.0, 0.0] limits (It can't rotate)
    target_pose = [0.4, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]
    assert not robot_state.differential_ik(group_name, target_pose)


def test_no_tcp_link():
    """Test the Robot class with no gripper and tcp_link."""
    robot_model = load_robot_model(
        FILE_PATH / ".." / "robots" / "acrobot" / "configs.toml"
    )
    group_name = "arm"
    assert robot_model.base_link == "universe"
    assert robot_model[group_name].tcp_link_name is None
    assert robot_model[group_name].joints == ["elbow"]
    assert robot_model[group_name].named_states["home"] == [0.0]


def test_robot():
    """Test the Robot class."""
    robot_model = load_robot_model(
        FILE_PATH / ".." / "robots" / "fr3_robotiq" / "configs.toml"
    )
    assert robot_model.base_link == "fr3_link0"
    group_name = "arm"
    assert robot_model.groups[group_name].tcp_link_name == "tcp_link"
    assert robot_model.groups[group_name].joints == [
        "fr3_joint1",
        "fr3_joint2",
        "fr3_joint3",
        "fr3_joint4",
        "fr3_joint5",
        "fr3_joint6",
        "fr3_joint7",
    ]

    # Test non-existing base link
    with pytest.raises(MissingBaseLinkError):
        robot_model = load_robot_model(FILE_PATH / "non_existing_base_link.toml")

    # Test non-existing group joint
    with pytest.raises(MissingJointError):
        robot_model = load_robot_model(FILE_PATH / "non_existing_group_joint.toml")


ROBOTS = ["fr3_robotiq", "kinova", "ur5e"]


@pytest.mark.parametrize("robot_name", ROBOTS)
def test_ik(robot_name):
    """Test the ik function."""
    robot_model = load_robot_model(
        FILE_PATH / ".." / "robots" / robot_name / "configs.toml",
    )
    # TODO: Add a loop to check for 100 different poses
    # Use fk to check if the pose is actually same as the input one
    group_name = "arm"
    robot_state = RobotState.from_named_state(robot_model, group_name, "home")
    assert robot_state.clone().differential_ik(
        group_name,
        [0.2, 0.2, 0.2, 1.0, 0.0, 0.0, 0.0],
    )
    ik_solver = IKSolver(
        robot_model.model_filename,
        robot_model.base_link,
        robot_model[group_name].tcp_link_name,
    )
    target_joint_positions = ik_solver.solve(
        [0.2, 0.2, 0.2, 1.0, 0.0, 0.0, 0.0],
        robot_state.actuated_qpos,
    )
    assert target_joint_positions is not None

    # Outside workspace should fail
    assert not robot_state.clone().differential_ik(
        group_name,
        [2.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0],
    )
    ik_solver = IKSolver(
        robot_model.model_filename,
        robot_model.base_link,
        robot_model[group_name].tcp_link_name,
    )
    target_joint_positions = ik_solver.solve(
        [2.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0],
        robot_state.actuated_qpos,
    )
    assert target_joint_positions is None


def test_robot_descriptions():
    """Test using robot-descriptions.py package."""
    with pytest.raises(RobotDescriptionNotFoundError):
        load_robot_model(
            FILE_PATH / "non_existing_robot_descriptions.toml",
        )
    load_robot_model(
        FILE_PATH / "panda_configs.toml",
    )

    load_robot_model(
        FILE_PATH / "panda_mj_configs.toml",
    )


def test_acceleration_limits():
    """Test the Robot class with missing joint acceleration limits."""
    with pytest.raises(MissingAccelerationLimitError):
        load_robot_model(
            FILE_PATH / "extra_acceleration_joint.toml",
        )
    with pytest.raises(MissingAccelerationLimitError):
        load_robot_model(
            FILE_PATH / "missing_acceleration_joint.toml",
        )
