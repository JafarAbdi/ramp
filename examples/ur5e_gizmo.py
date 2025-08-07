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
from ramp.visualizers import ViserVisualizer
from ramp.robot_model import create_geometry_object
from ramp.trajectory_smoothing import generate_time_optimal_trajectory
import hppfcl

setup_logging()

FILE_PATH = pathlib.Path(__file__).parent
robot_model = load_robot_model(FILE_PATH / ".." / "robots" / "ur5e" / "configs.toml")

visualizer = ViserVisualizer(robot_model)

start_state = RobotState(robot_model)

while True:
    pass
