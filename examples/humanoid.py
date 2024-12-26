"""Example of using the MotionPlanner functionality."""

import pathlib

import numpy as np
import pinocchio as pin

from ramp.robot import Robot, RobotState
from ramp.motion_planner import MotionPlanner
from ramp.visualizer import Visualizer

robot = Robot(pathlib.Path(f"robots/unitree_h1/configs.toml"))
visualize = Visualizer(robot)

LEFT_GROUP_NAME = "left_arm"
RIGHT_GROUP_NAME = "right_arm"

robot_state = RobotState(
    robot.robot_model,
    robot.robot_model.named_state(LEFT_GROUP_NAME, "home"),
)
visualize.robot_state(robot_state)

planner = MotionPlanner(robot, LEFT_GROUP_NAME)
if path := planner.plan(
    robot_state,
    [0.5, 0.5, 0.5, 0.5],
    timeout=5.0,
):
    visualize.robot_trajectory(path)

input("Press Enter to continue...")
planner = MotionPlanner(robot, RIGHT_GROUP_NAME)
if path := planner.plan(
    robot_state,
    [-0.5, -0.5, -0.5, -0.5],
    timeout=5.0,
):
    visualize.robot_trajectory(path)
