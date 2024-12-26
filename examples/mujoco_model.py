"""Example of using the differential_ik functionality."""

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
visualizer = Visualizer(robot)

group_name = "arm"
start_state = RobotState(robot.robot_model, [0.0])
goal_state = [3.14]
visualizer.robot_state(start_state)

input("Press Enter to continue...")
planner = MotionPlanner(robot, group_name)
if path := planner.plan(start_state, goal_state):
    # For visualizing the path parameterization - visualizer.robot_trajectory(path)
    trajectory = planner.parameterize(path)
    visualizer.robot_trajectory([t for t, _ in trajectory])
