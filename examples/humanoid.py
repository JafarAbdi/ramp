"""Example of using the MotionPlanner functionality."""

import sys
import pathlib

import numpy as np
import pinocchio as pin

from ramp.robot import Robot, RobotState
from ramp.motion_planner import MotionPlanner
from ramp.visualizer import Visualizer

robot = Robot(pathlib.Path(f"robots/unitree_h1/configs.toml"))

LEFT_GROUP_NAME = "left_arm"
RIGHT_GROUP_NAME = "right_arm"

robot_state = RobotState.from_named_state(
    robot.robot_model,
    LEFT_GROUP_NAME,
    "home",
)
visualize = False
if len(sys.argv) > 1 and sys.argv[1] == "visualize":
    visualize = True
    visualizer = Visualizer(robot)
    visualizer.robot_state(robot_state)

planner = MotionPlanner(robot.robot_model, LEFT_GROUP_NAME)
if path := planner.plan(
    robot_state,
    [0.5, 0.5, 0.5, 0.5],
    timeout=5.0,
):
    print(f"Found a path with {len(path)} waypoints")
    if visualize:
        visualizer.robot_trajectory(path)

input("Press Enter to continue...")
planner = MotionPlanner(robot.robot_model, RIGHT_GROUP_NAME)
if path := planner.plan(
    robot_state,
    [-0.5, -0.5, -0.5, -0.5],
    timeout=5.0,
):
    print(f"Found a path with {len(path)} waypoints")
    if visualize:
        visualizer.robot_trajectory(path)
