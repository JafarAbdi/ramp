"""Example of using the differential_ik functionality."""

import logging
import pathlib

from rich.logging import RichHandler

from ramp.robot import MotionPlanner, Robot, Visualizer

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
LOGGER = logging.getLogger(__name__)

robot = Robot(pathlib.Path("robots/acrobot/configs.toml"))
visualizer = Visualizer(robot)

start_state = [0.0, 0.0]
goal_state = [3.14, 3.14]
visualizer.robot_state(start_state)

input("Press Enter to continue...")
planner = MotionPlanner(robot)
if path := planner.plan(start_state, goal_state):
    trajectory = planner.parameterize(path)
    # For visualizing the path without parameterization visualizer.robot_trajectory(path)
    visualizer.robot_trajectory([t for t, _, _ in trajectory])

visualizer.robot_state(goal_state)
