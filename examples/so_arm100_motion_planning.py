"""Example of using the MotionPlanner functionality."""

import pathlib
import time

import numpy as np
import pinocchio as pin

from ramp import load_robot_model, RobotState, MotionPlanner, setup_logging
from ramp.compute_disable_collisions import (
    disable_collision_pair,
    disable_collision,
    adjacent_collisions,
)
from ramp.visualizer import Visualizer
from ramp.robot_model import create_geometry_object
from ramp.trajectory_smoothing import generate_time_optimal_trajectory
import hppfcl

setup_logging()

FILE_PATH = pathlib.Path(__file__).parent
group_name = "default"

robot_model = load_robot_model(
    "robot-descriptions::so_arm100_mj_description",
    acceleration_limits={
        "Rotation": 10.0,
        "Pitch": 10.0,
        "Elbow": 10.0,
        "Wrist_Pitch": 10.0,
        "Wrist_Roll": 10.0,
        "Jaw": 10.0,
    },
)
visualizer = Visualizer(robot_model)

start_state = RobotState(robot_model)
goal_state = RobotState(robot_model)
goal_state.add_object(
    create_geometry_object(
        "floor",
        hppfcl.Box(np.array([2.5, 2.5, 0.05])),
        pin.XYZQUATToSE3([0.0, 0.0, -0.025, 0.0, 0.0, 0.0, 1.0]),
    )
)
disable_collision(robot_model, adjacent_collisions(robot_model), verbose=True)
disable_collision_pair(robot_model, "Base", "floor")

planner = MotionPlanner(robot_model, group_name)
for _ in range(100):
    goal_state.randomize()
    if path := planner.plan(
        start_state,
        goal_state.actuated_qpos(),
        timeout=1.0,
    ):
        visualizer.robot_trajectory(path)
        start_state = goal_state.clone()
        input("Press Enter to continue...")
