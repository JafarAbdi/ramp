import pathlib

import numpy as np
import pinocchio
import pytest

import hppfcl
from ramp import load_robot_model, RobotState, MotionPlanner, setup_logging
from ramp.robot_model import create_geometry_object
from ramp.ik_solver import IKSolver
from ramp.trajectory_smoothing import generate_time_optimal_trajectory
from ramp.exceptions import (
    MissingBaseLinkError,
    MissingAccelerationLimitError,
    MissingJointError,
    RobotDescriptionNotFoundError,
)

FILE_PATH = pathlib.Path(__file__).parent
GROUP_NAME = "arm"

setup_logging()


def test_motion_planning():
    """Test motion planning interface."""
    robot_model = load_robot_model(FILE_PATH / ".." / "robots" / "rrr" / "configs.toml")
    planner = MotionPlanner(robot_model, GROUP_NAME)
    start_state = RobotState.from_named_state(robot_model, GROUP_NAME, "home")
    goal_state = start_state.clone()
    goal_state[GROUP_NAME] = [0.0, 1.0, 1.0]
    plan = planner.plan(start_state, goal_state)
    assert plan is not None, "Expected a plan to be found"
    trajectory = generate_time_optimal_trajectory(robot_model, GROUP_NAME, plan)
    assert trajectory is not None, "Expected a trajectory to be found"

    # RRR with planar base
    robot_model = load_robot_model(
        FILE_PATH / ".." / "robots" / "rrr" / "planar_configs.toml"
    )
    planner = MotionPlanner(robot_model, GROUP_NAME)
    start_state = RobotState.from_named_state(robot_model, GROUP_NAME, "home")
    start_state.add_object(
        create_geometry_object(
            "capsule",
            hppfcl.Capsule(0.1, 0.4),
            pinocchio.SE3(
                pinocchio.Quaternion(0.707, 0.707, 0.0, 0.0),
                np.asarray([0.475, 0.0, 0.5]),
            ),
        )
    )
    goal_state = start_state.clone()
    goal_state[GROUP_NAME] = [1.0, -0.5, 1.57, 0.5, 0.25, 0.1]
    plan = planner.plan(start_state, goal_state, timeout=5.0)
    assert plan is not None, "Expected a plan to be found"

    # RRR with floating base
    robot_model = load_robot_model(
        FILE_PATH / ".." / "robots" / "rrr" / "floating_configs.toml"
    )
    planner = MotionPlanner(robot_model, GROUP_NAME)
    start_state = RobotState.from_named_state(robot_model, GROUP_NAME, "home")
    start_state.add_object(
        create_geometry_object(
            "capsule",
            hppfcl.Capsule(0.1, 0.4),
            pinocchio.SE3(
                pinocchio.Quaternion(0.707, 0.707, 0.0, 0.0),
                np.asarray([0.475, 0.0, 0.5]),
            ),
        )
    )
    goal_state = start_state.clone()
    goal_state[GROUP_NAME] = [1.0, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.25, 0.1]
    plan = planner.plan(start_state, goal_state, timeout=5.0)
    assert plan is not None, "Expected a plan to be found"
