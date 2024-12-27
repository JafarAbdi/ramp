"""Example of using the differential_ik functionality."""

import sys
import logging
import pathlib

from rich.logging import RichHandler

from ramp.robot import Robot, RobotState
from ramp.motion_planner import MotionPlanner
from ramp.visualizer import Visualizer

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
LOGGER = logging.getLogger(__name__)

robot = Robot(pathlib.Path("robots/acrobot/configs.toml"))

group_name = "arm"
start_state = RobotState.from_actuated_qpos(robot.robot_model, [0.0])
goal_state = [3.14]
visualize = False
if len(sys.argv) > 1 and sys.argv[1] == "visualize":
    visualize = True

if visualize:
    visualizer = Visualizer(robot)
    visualizer.robot_state(start_state)

input("Press Enter to continue...")
planner = MotionPlanner(robot.robot_model, group_name)
if path := planner.plan(start_state, goal_state):
    # For visualizing the path parameterization - visualizer.robot_trajectory(path)
    trajectory = planner.parameterize(path)
    if visualize:
        visualizer.robot_trajectory([t for t, _ in trajectory])
