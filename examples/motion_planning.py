"""Example of using the MotionPlanner functionality."""

import pathlib
import sys

import numpy as np
import pinocchio as pin
import hppfcl

from ramp import (
    load_robot_model,
    RobotState,
    MotionPlanner,
    MeshcatVisualizer,
    setup_logging,
)
from ramp.robot_model import create_geometry_object
import logging

LOGGER = logging.getLogger(__name__)
setup_logging()

group_name = "arm"
robots = [
    ("panda/configs", [0.0, 0.25, 0.0, -1.25, 0.0, 1.5, 0.785]),
    ("panda_mj/configs", [0.0, 0.25, 0.0, -1.25, 0.0, 1.5, 0.785]),
    ("fr3/configs", [0.0, 0.25, 0.0, -1.25, 0.0, 1.5, 0.0]),
    ("ur5e/configs", [0.0, -1.57, 0.0, -1.57, -1.57, 0.0]),
    ("rrr/configs", [0.0, 2.0, 1.4]),
    ("rrr/floating_configs", [1.0, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.25, 0.1]),
    ("rrr/owen_configs", [1.0, 0.5, 1.0, 0.0, 0.0, 0.707, 0.707, 0.5, 0.25, 0.1]),
    ("rrr/vana_configs", [1.0, 0.5, 1.0, 0.0, 0.0, 0.707, 0.707, 0.5, 0.25, 0.1]),
    ("rrr/vana_owen_configs", [1.0, 0.5, 1.0, 0.0, 0.0, 0.707, 0.707, 0.5, 0.25, 0.1]),
    ("rrr/planar_configs", [1.0, -0.5, 1.57, 0.5, 0.25, 0.1]),
    ("rrr/reeds_shepp_configs", [1.0, -0.5, 1.57, 0.5, 0.25, 0.1]),
    ("rrr/dubins_configs", [1.0, -0.5, 1.57, 0.5, 0.25, 0.1]),
]

visualize = False
if len(sys.argv) > 1 and sys.argv[1] == "visualize":
    visualize = True

for robot_path, goal_state in robots:
    LOGGER.info(f"Planning for robot {robot_path}.")
    robot_model = load_robot_model(pathlib.Path(f"robots/{robot_path}.toml"))
    if visualize:
        input(
            f"Press Enter to continue with robot {robot_path} and a collision object in the scene..."
        )
        visualizer = MeshcatVisualizer(robot_model)

    start_state = RobotState.from_named_state(robot_model, group_name, "home")
    start_state.add_object(
        create_geometry_object(
            "capsule",
            hppfcl.Capsule(0.1, 0.4),
            pin.SE3(
                pin.Quaternion(0.707, 0.707, 0.0, 0.0), np.asarray([0.475, 0.0, 0.5])
            ),
        )
    )
    if visualize:
        visualizer.robot_state(start_state)
        input("Press Enter to plan a path to the goal state...")

    planner = MotionPlanner(robot_model, group_name)
    if path := planner.plan(start_state, goal_state, timeout=5.0):
        LOGGER.info(f"Found a path with {len(path)} waypoints.")
        if visualize:
            visualizer.robot_trajectory(path)
