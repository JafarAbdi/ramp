"""A script to print information about a model."""

import logging
import sys

from ramp import MeshcatVisualizer, load_robot_model, setup_logging

setup_logging()

LOGGER = logging.getLogger(__name__)


def main():
    """Main function."""
    if len(sys.argv) != 2:
        LOGGER.error(f"Usage: {sys.argv[0]} <model_filename>")
        sys.exit(1)

    robot_model = load_robot_model(sys.argv[1])
    LOGGER.info(f"Joints: {list(robot_model.model.names)}")
    LOGGER.info(f"Frames: {[frame.name for frame in robot_model.model.frames]}")

    visualizer = MeshcatVisualizer(robot_model)
    visualizer.meshcat_visualizer.displayCollisions(visibility=True)


if __name__ == "__main__":
    main()
