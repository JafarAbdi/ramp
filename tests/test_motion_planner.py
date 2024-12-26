import pathlib

import numpy as np
import pinocchio
import pytest

from ramp.robot import Robot, RobotState, GroupState
from ramp.constants import GROUP_NAME
from ramp.ik_solver import IKSolver
from ramp.exceptions import (
    MissingBaseLinkError,
    MissingAccelerationLimitError,
    MissingJointError,
    RobotDescriptionNotFoundError,
)
from ramp.motion_planner import MotionPlanner

FILE_PATH = pathlib.Path(__file__).parent


def test_motion_planning():
    """Test motion planning interface."""
    robot = Robot(FILE_PATH / ".." / "robots" / "rrr" / "configs.toml")
    planner = MotionPlanner(robot, GROUP_NAME)
    start_state = RobotState(
        robot.robot_model, robot.robot_model[GROUP_NAME].named_states["home"]
    )
    goal_state = RobotState(robot.robot_model, [0.0, 1.0, 1.0])
    plan = planner.plan(start_state, goal_state)
    assert plan is not None, "Expected a plan to be found"
    trajectory = planner.parameterize(plan)
    assert trajectory is not None, "Expected a trajectory to be found"

    # RRR with planar base
    robot = Robot(FILE_PATH / ".." / "robots" / "rrr" / "planar_configs.toml")
    robot.add_object(
        "capsule",
        pinocchio.GeometryObject.CreateCapsule(0.1, 0.4),
        pinocchio.SE3(
            pinocchio.Quaternion(0.707, 0.707, 0.0, 0.0),
            np.asarray([0.475, 0.0, 0.5]),
        ),
    )
    planner = MotionPlanner(robot, GROUP_NAME)
    start_state = RobotState(
        robot.robot_model, robot.robot_model[GROUP_NAME].named_states["home"]
    )
    goal_state = RobotState(robot.robot_model, [1.0, -0.5, 1.57, 0.5, 0.25, 0.1])
    plan = planner.plan(start_state, goal_state, timeout=5.0)
    assert plan is not None, "Expected a plan to be found"

    # RRR with floating base
    robot = Robot(FILE_PATH / ".." / "robots" / "rrr" / "floating_configs.toml")
    robot.add_object(
        "capsule",
        pinocchio.GeometryObject.CreateCapsule(0.1, 0.4),
        pinocchio.SE3(
            pinocchio.Quaternion(0.707, 0.707, 0.0, 0.0),
            np.asarray([0.475, 0.0, 0.5]),
        ),
    )
    planner = MotionPlanner(robot, GROUP_NAME)
    start_state = RobotState(
        robot.robot_model, robot.robot_model[GROUP_NAME].named_states["home"]
    )
    goal_state = RobotState(
        robot.robot_model, [1.0, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.25, 0.1]
    )
    plan = planner.plan(start_state, goal_state, timeout=5.0)
    assert plan is not None, "Expected a plan to be found"
