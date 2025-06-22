"""Example of using the MotionPlanner functionality."""

import pathlib
import sys

import numpy as np
import pinocchio as pin
import hppfcl

from ramp import load_robot_model, RobotState, MotionPlanner, Visualizer, setup_logging
from ramp.robot_model import create_geometry_object

setup_logging()

group_name = "default"

visualize = False
if len(sys.argv) > 1 and sys.argv[1] == "visualize":
    visualize = True

rotation = pin.Quaternion(0.707, 0.707, 0.0, 0.0).normalize()
goal_state = [0.4, 0.4, 0.4, rotation.x, rotation.y, rotation.z, rotation.w]

robot_model = load_robot_model(
    pathlib.Path("robots/floating.urdf"), {"floating_joint": "vana"}
)
if visualize:
    visualizer = Visualizer(robot_model)

start_state = RobotState.from_actuated_qpos(
    robot_model, [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0]
)

for i in range(3):
    for j in range(3):
        for k in range(3):
            geometry_object = pin.GeometryObject(
                f"sphere_{i}x{j}x{k}",
                0,
                pin.SE3(
                    pin.Quaternion.Identity(),
                    np.asarray([i * 0.25, j * 0.25, k * 0.25]),
                ),
                hppfcl.Sphere(0.05),
            )
            start_state.add_object(geometry_object)
if visualize:
    visualizer.robot_state(start_state)
    input("Press Enter to plan a path to the goal state...")

planner = MotionPlanner(robot_model, group_name)
if path := planner.plan(start_state, goal_state, timeout=10.0):
    print(f"Found a path with {len(path)} waypoints.")
    if visualize:
        visualizer.robot_trajectory(path)
