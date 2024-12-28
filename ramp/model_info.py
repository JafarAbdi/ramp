"""A script to print information about a model."""

import logging
import pathlib
import sys

import pinocchio

from ramp import setup_logging

setup_logging()

LOGGER = logging.getLogger(__name__)


def main():
    """Main function."""
    if len(sys.argv) != 2:
        LOGGER.error(f"Usage: {sys.argv[0]} <model_filename>")
        sys.exit(1)

    model_filename = pathlib.Path(sys.argv[1])

    if not model_filename.exists():
        LOGGER.error(f"Model file {model_filename} does not exist")
        sys.exit(1)

    if model_filename.suffix != ".xml":
        LOGGER.error(f"Only mjcf files are supported! Input file {model_filename}")
        sys.exit(1)

    LOGGER.info(f"Loading model from {model_filename}")
    models: tuple[pinocchio.Model, pinocchio.Model, pinocchio.Model] = (
        pinocchio.shortcuts.buildModelsFromMJCF(
            model_filename,
            verbose=True,
        )
    )
    model, visual_model, collision_model = models
    LOGGER.info(f"Joints: {list(model.names)}")
    LOGGER.info(f"Frames: {[frame.name for frame in model.frames]}")


if __name__ == "__main__":
    main()
