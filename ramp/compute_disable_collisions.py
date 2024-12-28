"""Compute the disable collisions for a given model."""

import logging
import pathlib
import sys

import pinocchio
from rich.logging import RichHandler

from ramp.pinocchio_utils import load_models

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
LOGGER = logging.getLogger(__name__)


def get_collision_pair_names(
    model: pinocchio.Model,
    collision_model: pinocchio.GeometryModel,
    pair: pinocchio.CollisionPair,
) -> tuple[str, str]:
    """Get the collision pair names."""
    frame_name1 = model.frames[
        collision_model.geometryObjects[pair.first].parentFrame
    ].name
    frame_name2 = model.frames[
        collision_model.geometryObjects[pair.second].parentFrame
    ].name
    return (frame_name1, frame_name2)


class ComputeDisableCollisions:
    """A class to compute the disable collisions for a model."""

    def __init__(self, model_filename):
        """Initialize the class.

        Args:
            model_filename: The model filename.
        """
        (model_filename, models) = load_models(model_filename, {})
        self.model: pinocchio.Model = models[0]
        self.visual_model: pinocchio.GeometryModel = models[1]
        self.collision_model: pinocchio.GeometryModel = models[2]
        self.collision_model.addAllCollisionPairs()

    def adjacents(self) -> list[tuple[str, str]]:
        """Get the adjacent links."""

        def get_body_parent(frame: pinocchio.Frame) -> pinocchio.Frame:
            """Get the body parent."""
            while (parent_frame := self.model.frames[frame.parentFrame]).type not in [
                pinocchio.BODY,
            ] and frame.name != "universe":
                frame = parent_frame
            return parent_frame

        adjacent_pairs = set()
        for geometry_object in self.collision_model.geometryObjects:
            current_frame = self.model.frames[geometry_object.parentFrame]
            parent_frame = get_body_parent(current_frame)
            if parent_frame.name == "universe" or current_frame.name == "universe":
                continue
            adjacent_pairs.add((parent_frame.name, current_frame.name))
        return list(adjacent_pairs)

    def check_collision(self, qpos):
        """Check the collision for a given configuration.

        Args:
            qpos: The configuration.

        Returns:
            The collision pairs.
        """
        data = self.model.createData()
        collision_data = self.collision_model.createData()
        pinocchio.computeCollisions(
            self.model,
            data,
            self.collision_model,
            collision_data,
            qpos,
            stop_at_first_collision=False,
        )
        pairs = {}
        for k in range(len(self.collision_model.collisionPairs)):
            cr = collision_data.collisionResults[k]
            cp = self.collision_model.collisionPairs[k]
            if cr.isCollision():
                pairs[
                    get_collision_pair_names(self.model, self.collision_model, cp)
                ] = cp
        return pairs

    def disable_collision(self, disable_collisions: list[tuple[str, str]]):
        """Disable the collisions for the given pairs.

        Args:
            disable_collisions: The pairs to disable.
        """
        srdf = """
<robot name="disable_collisions">
    {disable_collisions}
</robot>
"""
        LOGGER.info(
            f"Number of collision pairs before removal: {len(self.collision_model.collisionPairs)}",
        )
        pinocchio.removeCollisionPairsFromXML(
            self.model,
            self.collision_model,
            srdf.format(disable_collisions="\n".join(disable_collisions)),
            verbose=True,
        )

        # TODO: Why this is not working???
        # > for pair in default_pairs.values():
        # >   compute_disable_collisions.collision_model.removeCollisionPair(pair)
        LOGGER.info(
            f"Number of collision pairs after removal: {len(self.collision_model.collisionPairs)}",
        )


def main():
    """Main function."""
    if len(sys.argv) != 2:
        LOGGER.error(f"Usage: {sys.argv[0]} <model_filename>")
        sys.exit(1)

    model_filename = pathlib.Path(sys.argv[1])

    if not model_filename.exists():
        LOGGER.error(f"Model file {model_filename} does not exist")
        sys.exit(1)

    LOGGER.info(f"Loading model from {model_filename}")
    compute_disable_collisions = ComputeDisableCollisions(model_filename)

    # DISABLE "DEFAULT" COLLISIONS
    # Disable all collision checks that occur when the robot is started in its default state
    default_pairs = compute_disable_collisions.check_collision(
        pinocchio.neutral(compute_disable_collisions.model),
    )
    disable_collisions = []
    for first, second in default_pairs:
        disable_collisions.append(
            f'<disable_collisions link1="{first}" link2="{second}" reason="Default"/>',
        )
    LOGGER.info("\n".join(disable_collisions))
    compute_disable_collisions.disable_collision(disable_collisions)

    # DISABLE ALL "ADJACENT" LINK COLLISIONS
    disable_collisions = []
    for first, second in compute_disable_collisions.adjacents():
        disable_collisions.append(
            f'<disable_collisions link1="{first}" link2="{second}" reason="Adjacent"/>',
        )
    LOGGER.info("\n".join(disable_collisions))
    compute_disable_collisions.disable_collision(disable_collisions)
    # "ALWAYS" IN COLLISION: Compute the links that are always in collision
    # "NEVER" IN COLLISION: Get the pairs of links that are never in collision
    for pair in compute_disable_collisions.collision_model.collisionPairs:
        LOGGER.info(
            f"Collision pair: {get_collision_pair_names(compute_disable_collisions.model, compute_disable_collisions.collision_model, pair)}",
        )

    LOGGER.info("Done!")


if __name__ == "__main__":
    main()
