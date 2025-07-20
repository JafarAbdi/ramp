"""Example of using the MotionPlanner functionality."""

import sys
import pathlib


from ramp import load_robot_model, RobotState, MotionPlanner, Visualizer

robot_model = load_robot_model(pathlib.Path("robots/unitree_h1/configs.toml"))

LEFT_GROUP_NAME = "left_arm"
RIGHT_GROUP_NAME = "right_arm"

robot_state = RobotState.from_named_state(
    robot_model,
    LEFT_GROUP_NAME,
    "home",
)
visualize = False
if len(sys.argv) > 1 and sys.argv[1] == "visualize":
    visualize = True
    visualizer = Visualizer(robot_model)
    visualizer.robot_state(robot_state)

planner = MotionPlanner(robot_model, LEFT_GROUP_NAME)
if path := planner.plan(
    robot_state,
    [0.5, 0.5, 0.5, 0.5],
    timeout=5.0,
):
    print(f"Found a path with {len(path)} waypoints")
    if visualize:
        visualizer.robot_trajectory(path)

if visualize:
    input("Press Enter to continue...")

planner = MotionPlanner(robot_model, RIGHT_GROUP_NAME)
if path := planner.plan(
    robot_state,
    [-0.5, -0.5, -0.5, -0.5],
    timeout=5.0,
):
    print(f"Found a path with {len(path)} waypoints")
    if visualize:
        visualizer.robot_trajectory(path)
