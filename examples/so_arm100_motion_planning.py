"""Example of using the MotionPlanner functionality."""

import pathlib
import sys
import time

import numpy as np
import pinocchio as pin

from ramp import load_robot_model, RobotState, MotionPlanner, setup_logging
from ramp.compute_disable_collisions import (
    disable_collision_pair,
    disable_collision,
    adjacent_collisions,
    default_collisions,
)
from ramp.mujoco_interface import MuJoCoHardwareInterface
from ramp.robot_model import create_geometry_object
from ramp.trajectory_smoothing import generate_time_optimal_trajectory
import hppfcl
import pinocchio as pin

setup_logging()

FILE_PATH = pathlib.Path(__file__).parent
group_name = "default"

robot_model = load_robot_model(
    FILE_PATH
    / ".."
    / "external"
    / "mujoco_menagerie"
    / "trs_so_arm100"
    / "so_arm100.xml",
    acceleration_limits={
        "Rotation": 10.0,
        "Pitch": 10.0,
        "Elbow": 10.0,
        "Wrist_Pitch": 10.0,
        "Wrist_Roll": 10.0,
        "Jaw": 10.0,
    },
)

mj_interface = MuJoCoHardwareInterface(robot_model)
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
mj_interface.add_decorative_geometry(
    "floor", "box", (0.0, 0.0, -0.025), (1.25, 1.25, 0.025)
)

goal_state = RobotState(robot_model)
goal_state.randomize()
planner = MotionPlanner(robot_model, group_name)
for _ in range(100):
    goal_state.randomize()
    if path := planner.plan(
        mj_interface.read(),
        goal_state,
        timeout=5.0,
    ):
        optimized_path = generate_time_optimal_trajectory(
            robot_model, group_name, path, 0.01
        )
        t = 0.0
        for rs, time_from_start in optimized_path:
            mj_interface.write(rs)
            time.sleep(time_from_start - t)
            t = time_from_start
