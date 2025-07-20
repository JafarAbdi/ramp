"""A script to print information about a model."""

import logging
import pathlib
import sys

from ramp import Visualizer, load_robot_model, setup_logging
from ramp.pinocchio_utils import get_robot_description_path
from ramp.constants import ROBOT_DESCRIPTION_PREFIX

setup_logging()

LOGGER = logging.getLogger(__name__)


def main():
    """Main function."""
    if len(sys.argv) != 2:
        LOGGER.error(f"Usage: {sys.argv[0]} <model_filename>")
        sys.exit(1)

    robot_description_path = (
        get_robot_description_path(sys.argv[1])
        if sys.argv[1].startswith(ROBOT_DESCRIPTION_PREFIX)
        else pathlib.Path(sys.argv[1])
    )
    robot_model = load_robot_model(robot_description_path)
    LOGGER.info(f"Joints: {list(robot_model.model.names)}")
    LOGGER.info(f"Frames: {[frame.name for frame in robot_model.model.frames]}")

    visualizer = Visualizer(robot_model)
    visualizer.meshcat_visualizer.displayCollisions(visibility=True)


if __name__ == "__main__":
    main()
