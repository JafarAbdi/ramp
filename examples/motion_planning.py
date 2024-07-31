"""Example of using the MotionPlanner functionality."""

import pathlib

import numpy as np
import pinocchio as pin

from ramp.robot import Robot
from ramp.motion_planner import MotionPlanner
from ramp.visualizer import Visualizer

robots = [
    ("panda/configs", [0.0, 0.25, 0.0, -1.25, 0.0, 1.5, 0.785, 0.0]),
    ("panda_mj/configs", [0.0, 0.25, 0.0, -1.25, 0.0, 1.5, 0.785, 0.0]),
    ("fr3_robotiq/configs", [0.0, 0.25, 0.0, -1.25, 0.0, 1.5, 0.0, 0.0]),
    ("ur5e/configs", [0.0, -1.57, 0.0, -1.57, -1.57, 0.0, 0.0]),
    ("kinova/configs", [0.0, 0.0, -3.14, -2.5, 0.0, 0.0, 1.57, 0.0]),
    ("rrr/configs", [0.0, 2.0, 1.4]),
    ("rrr/floating_configs", [1.0, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.25, 0.1]),
    ("rrr/planar_configs", [1.0, -0.5, 1.57, 0.5, 0.25, 0.1]),
]

for robot_path, goal_state in robots:
    input(
        f"Press Enter to continue with robot {robot_path} and a collision object in the scene..."
    )
    robot = Robot(pathlib.Path(f"robots/{robot_path}.toml"))
    visualize = Visualizer(robot)
    start_state = robot.named_states["home"]
    visualize.robot_state(start_state)

    robot.add_object(
        "capsule",
        pin.GeometryObject.CreateCapsule(0.1, 0.4),
        pin.SE3(pin.Quaternion(0.707, 0.707, 0.0, 0.0), np.asarray([0.475, 0.0, 0.5])),
    )

    start_state = robot.named_states["home"]
    visualize.robot_state(start_state)

    planner = MotionPlanner(robot)
    if path := planner.plan(start_state, goal_state, timeout=5.0):
        visualize.robot_trajectory(path)
    visualize.robot_state(goal_state)
