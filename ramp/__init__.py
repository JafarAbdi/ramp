"""Init."""

import importlib.metadata
import logging
import os

from rich.logging import RichHandler

from .motion_planner import MotionPlanner
from .robot_model import RobotModel, load_robot_model, CasADiRobot
from .robot_state import RobotState
from .visualizer import Visualizer


def setup_logging():
    """Setup logging for the package."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )


LOGGER = logging.getLogger(__name__)

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError as e:  # pragma: no cover
    LOGGER.warning(f"Could not determine version of {__name__}\n{e!s}", stacklevel=2)
    __version__ = "unknown"

__all__ = [
    "MotionPlanner",
    "Visualizer",
    "RobotState",
    "RobotModel",
    "CasADiRobot",
    "load_robot_model",
]
