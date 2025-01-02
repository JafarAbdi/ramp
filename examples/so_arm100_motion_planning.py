"""Example of using the MotionPlanner functionality."""

import pathlib
import sys

import numpy as np
import pinocchio as pin

from ramp import load_robot_model, RobotState, MotionPlanner, Visualizer, setup_logging
from ramp.compute_disable_collisions import disable_collision, adjacent_collisions

setup_logging()

FILE_PATH = pathlib.Path(__file__).parent
group_name = "default"
HOME = [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]

visualize = False
if len(sys.argv) > 1 and sys.argv[1] == "visualize":
    visualize = True

robot_model = load_robot_model(
    FILE_PATH
    / ".."
    / "external"
    / "mujoco_menagerie"
    / "trs_so_arm100"
    / "so_arm100.xml"
)
disable_collision(robot_model, adjacent_collisions(robot_model), verbose=True)

if visualize:
    visualizer = Visualizer(robot_model)

start_state = RobotState.from_actuated_qpos(robot_model, HOME)
if visualize:
    visualizer.robot_state(start_state)

planner = MotionPlanner(robot_model, group_name)
while True:
    input("Press Enter to plan a path to the goal state...")
    if path := planner.plan(
        start_state, RobotState.from_random(robot_model).actuated_qpos, timeout=5.0
    ):
        if visualize:
            visualizer.robot_trajectory(path)
