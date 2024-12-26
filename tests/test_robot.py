"""Test the Robot class."""

import pathlib

import numpy as np
import pinocchio
import pytest

from ramp.robot import (
    Robot,
)
from ramp.constants import GROUP_NAME
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
    robot = Robot(FILE_PATH / ".." / "robots" / "planar_rrr" / "configs.toml")


def test_no_gripper():
    """Test the Robot class with no gripper and tcp_link."""
    robot = Robot(FILE_PATH / ".." / "robots" / "rrr" / "configs.toml")
    assert robot.base_link == "base_link"
    assert robot.groups[GROUP_NAME].tcp_link_name == "end_effector"
    assert robot.groups[GROUP_NAME].joints == ["joint1", "joint2", "joint3"]
    assert robot.groups[GROUP_NAME].gripper is None
    target_pose = [0.4, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0]
    # Using IK
    ik_solver = IKSolver(
        robot.model_filename, robot.base_link, robot.groups[GROUP_NAME].tcp_link_name
    )
    target_joint_positions = ik_solver.solve(target_pose, robot.named_states["home"])
    assert target_joint_positions is not None

    # Using differential-ik
    target_joint_positions = robot.differential_ik(
        target_pose,
        robot.named_states["home"],
    )
    assert target_joint_positions is not None
    pose = robot.get_frame_pose(
        target_joint_positions,
        robot.groups[GROUP_NAME].tcp_link_name,
    )
    assert np.allclose(target_pose, pinocchio.SE3ToXYZQUAT(pose), atol=1e-3)

    # Should fail, rrr robot has end_effector_joint as revolute joint with [0.0, 0.0] limits (It can't rotate)
    target_pose = [0.4, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]
    target_joint_positions = robot.differential_ik(
        target_pose,
        robot.named_states["home"],
    )
    assert target_joint_positions is None


def test_no_gripper_and_tcp_link():
    """Test the Robot class with no gripper and tcp_link."""
    robot = Robot(FILE_PATH / ".." / "robots" / "acrobot" / "configs.toml")
    assert robot.base_link == "universe"
    assert robot.groups[GROUP_NAME].tcp_link_name is None
    assert robot.groups[GROUP_NAME].joints == ["elbow"]
    assert robot.groups[GROUP_NAME].gripper is None
    assert robot.named_states["home"] == [0.0]


def test_robot():
    """Test the Robot class."""
    robot = Robot(FILE_PATH / ".." / "robots" / "fr3_robotiq" / "configs.toml")
    assert robot.base_link == "fr3_link0"
    assert robot.groups[GROUP_NAME].tcp_link_name == "tcp_link"
    assert robot.groups[GROUP_NAME].joints == [
        "fr3_joint1",
        "fr3_joint2",
        "fr3_joint3",
        "fr3_joint4",
        "fr3_joint5",
        "fr3_joint6",
        "fr3_joint7",
    ]
    gripper = robot.groups[GROUP_NAME].gripper
    assert gripper.actuated_joint == "robotiq_85_left_inner_knuckle_joint"
    assert robot.joint_names == [
        "fr3_joint1",
        "fr3_joint2",
        "fr3_joint3",
        "fr3_joint4",
        "fr3_joint5",
        "fr3_joint6",
        "fr3_joint7",
        "robotiq_85_left_inner_knuckle_joint",
    ]

    # Test non-existing base link
    with pytest.raises(MissingBaseLinkError):
        robot = Robot(FILE_PATH / "non_existing_base_link.toml")

    # Test non-existing group joint
    with pytest.raises(MissingJointError):
        robot = Robot(FILE_PATH / "non_existing_group_joint.toml")


ROBOTS = ["fr3_robotiq", "kinova", "ur5e"]


@pytest.mark.parametrize("robot_name", ROBOTS)
def test_ik(robot_name):
    """Test the ik function."""
    robot = Robot(
        pathlib.Path(
            FILE_PATH / ".." / "robots" / robot_name / "configs.toml",
        ),
    )
    # TODO: Add a loop to check for 100 different poses
    # Use fk to check if the pose is actually same as the input one
    target_joint_positions = robot.differential_ik(
        [0.2, 0.2, 0.2, 1.0, 0.0, 0.0, 0.0],
        robot.named_states["home"],
    )
    assert target_joint_positions is not None
    ik_solver = IKSolver(
        robot.model_filename, robot.base_link, robot.groups[GROUP_NAME].tcp_link_name
    )
    target_joint_positions = ik_solver.solve(
        [0.2, 0.2, 0.2, 1.0, 0.0, 0.0, 0.0],
        robot.named_states["home"][:-1],
    )
    assert target_joint_positions is not None

    # Outside workspace should fail
    target_joint_positions = robot.differential_ik(
        [2.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0],
        robot.named_states["home"],
    )
    assert target_joint_positions is None
    ik_solver = IKSolver(
        robot.model_filename, robot.base_link, robot.groups[GROUP_NAME].tcp_link_name
    )
    target_joint_positions = ik_solver.solve(
        [2.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0],
        robot.named_states["home"][:-1],
    )
    assert target_joint_positions is None


def strip_whitespace(text):
    """Given a text remove all the white spaces from the text."""
    return "".join(line.strip() for line in text.splitlines() if line.strip())


def test_robot_descriptions():
    """Test using robot-descriptions.py package."""
    with pytest.raises(RobotDescriptionNotFoundError):
        Robot(
            pathlib.Path(
                FILE_PATH / "non_existing_robot_descriptions.toml",
            ),
        )
    Robot(
        pathlib.Path(
            FILE_PATH / "panda_configs.toml",
        ),
    )

    Robot(
        pathlib.Path(
            FILE_PATH / "panda_mj_configs.toml",
        ),
    )


def test_acceleration_limits():
    """Test the Robot class with missing joint acceleration limits."""
    with pytest.raises(MissingAccelerationLimitError):
        Robot(
            pathlib.Path(
                FILE_PATH / "extra_acceleration_joint.toml",
            ),
        )
    with pytest.raises(MissingAccelerationLimitError):
        Robot(
            pathlib.Path(
                FILE_PATH / "missing_acceleration_joint.toml",
            ),
        )
