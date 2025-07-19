"""Compute the disable collisions for a given model."""

import enum
import itertools
import logging
import pathlib
import sys

import pinocchio
from rich.logging import RichHandler

from ramp.constants import SIZE_T_MAX
from ramp.robot_model import RobotModel, load_robot_model


class DisabledReason(enum.StrEnum):
    """The reason why the collision is disabled."""

    DEFAULT = "Default"
    ADJACENT = "Adjacent"
    ALWAYS = "Always"
    NEVER = "Never"


LOGGER = logging.getLogger(__name__)

SRDF = """
<robot name="disable_collisions">
{disable_collisions}
</robot>
"""
DISABLE_COLLISIONS_TAG = (
    '<disable_collisions link1="{link1}" link2="{link2}" reason="{reason}"/>'
)


def get_collision_pair_names(
    model: pinocchio.Model,
    collision_model: pinocchio.GeometryModel,
    pair: pinocchio.CollisionPair,
) -> tuple[str, str]:
    """Get the collision pair names."""
    frame_name1 = collision_model.geometryObjects[pair.first].name
    if collision_model.geometryObjects[pair.first].parentFrame != SIZE_T_MAX:
        frame_name1 = model.frames[
            collision_model.geometryObjects[pair.first].parentFrame
        ].name
    frame_name2 = collision_model.geometryObjects[pair.second].name
    if collision_model.geometryObjects[pair.second].parentFrame != SIZE_T_MAX:
        frame_name2 = model.frames[
            collision_model.geometryObjects[pair.second].parentFrame
        ].name
    return (frame_name1, frame_name2)


def disable_collision_pair(
    robot_model: RobotModel,
    link1: str,
    link2: str,
):
    """Disable the collision between two links.

    Args:
        robot_model: The robot model.
        link1: The first link.
        link2: The second link.

    Returns:
        The collision pair.
    """
    LOGGER.info(
        f"Number of collision pairs before removal: {len(robot_model.collision_model.collisionPairs)}",
    )
    link1_geometries = [
        geometry.name for geometry in robot_model.body_geometries(link1)
    ] or [link1]
    link2_geometries = [
        geometry.name for geometry in robot_model.body_geometries(link2)
    ] or [link2]
    for geometry1, geometry2 in itertools.product(link1_geometries, link2_geometries):
        robot_model.collision_model.removeCollisionPair(
            pinocchio.CollisionPair(
                robot_model.collision_model.getGeometryId(geometry1),
                robot_model.collision_model.getGeometryId(geometry2),
            ),
        )
    LOGGER.info(
        f"Number of collision pairs after removal: {len(robot_model.collision_model.collisionPairs)}",
    )


def disable_collision(
    robot_model: RobotModel,
    disable_collisions: list[tuple[str, str, DisabledReason]],
    *,
    verbose=False,
):
    """Disable the collisions for the given pairs.

    Args:
        robot_model: The robot model.
        disable_collisions: The pairs to disable.
        verbose: Whether to print the pairs that are being disabled.
    """
    LOGGER.info(
        f"Number of collision pairs before removal: {len(robot_model.collision_model.collisionPairs)}",
    )
    disable_collision_tags = [
        DISABLE_COLLISIONS_TAG.format(
            link1=link1,
            link2=link2,
            reason=reason,
        )
        for link1, link2, reason in disable_collisions
    ]
    if verbose:
        if not disable_collision_tags:
            LOGGER.info("No collision pairs to disable.")
            return
        disable_collision_formatted = "\n".join(disable_collision_tags)
        LOGGER.info(
            f"Disabling the following collision pairs:\n{disable_collision_formatted}",
        )
    pinocchio.removeCollisionPairsFromXML(
        robot_model.model,
        robot_model.collision_model,
        SRDF.format(disable_collisions="\n".join(disable_collision_tags)),
        verbose=True,
    )

    # TODO: Why this is not working??? Body name != geometry name
    # > for pair in default_pairs.values():
    # >   compute_disable_collisions.collision_model.removeCollisionPair(pair)
    LOGGER.info(
        f"Number of collision pairs after removal: {len(robot_model.collision_model.collisionPairs)}",
    )


def get_collision_pairs(
    robot_model: RobotModel,
    qpos,
) -> dict[tuple[str, str], pinocchio.CollisionPair]:
    """Check the collision for a given configuration and return the collision pairs.

    Args:
        robot_model: The robot model.
        qpos: The configuration.

    Returns:
        The collision pairs.
    """
    data = robot_model.model.createData()
    collision_data = robot_model.collision_model.createData()
    pinocchio.computeCollisions(
        robot_model.model,
        data,
        robot_model.collision_model,
        collision_data,
        qpos,
        stop_at_first_collision=False,
    )
    pairs = {}
    for k in range(len(robot_model.collision_model.collisionPairs)):
        cr = collision_data.collisionResults[k]
        cp = robot_model.collision_model.collisionPairs[k]
        if cr.isCollision():
            pairs[
                get_collision_pair_names(
                    robot_model.model,
                    robot_model.collision_model,
                    cp,
                )
            ] = cp
    return pairs


# DISABLE "DEFAULT" COLLISIONS
def default_collisions(
    robot_model: RobotModel,
) -> list[tuple[str, str, DisabledReason]]:
    """Disable the default collisions."""
    # Disable all collision checks that occur when the robot is started in its default state
    default_pairs = get_collision_pairs(
        robot_model,
        pinocchio.neutral(robot_model.model),
    )
    disable_collisions = []
    for first, second in default_pairs:
        disable_collisions.append(
            (
                first,
                second,
                DisabledReason.DEFAULT,
            ),
        )
    return disable_collisions


# DISABLE "ADJACENT" LINK COLLISIONS
def adjacent_collisions(
    robot_model: RobotModel,
) -> list[str, str, DisabledReason]:
    """Get the adjacent links."""

    def get_body_parent(frame: pinocchio.Frame) -> pinocchio.Frame:
        """Get the body parent."""
        while (
            parent_frame := robot_model.model.frames[frame.parentFrame]
        ).type not in [
            pinocchio.BODY,
        ] and frame.name != "universe":
            frame = parent_frame
        return parent_frame

    adjacent_pairs = set()
    for geometry_object in robot_model.collision_model.geometryObjects:
        if geometry_object.parentFrame == SIZE_T_MAX:
            continue
        current_frame = robot_model.model.frames[geometry_object.parentFrame]
        parent_frame = get_body_parent(current_frame)
        if parent_frame.name == "universe" or current_frame.name == "universe":
            continue
        adjacent_pairs.add(
            (parent_frame.name, current_frame.name, DisabledReason.ADJACENT),
        )
    return list(adjacent_pairs)


# TODO: "ALWAYS" IN COLLISION: Compute the links that are always in collision
# TODO: "NEVER" IN COLLISION: Get the pairs of links that are never in collision


def main():
    """Main function."""
    logging.basicConfig(
        level="NOTSET",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )
    if len(sys.argv) != 2:
        LOGGER.error(f"Usage: {sys.argv[0]} <model_filename>")
        sys.exit(1)

    model_filename = pathlib.Path(sys.argv[1])

    if not model_filename.exists():
        LOGGER.error(f"Model file {model_filename} does not exist")
        sys.exit(1)

    LOGGER.info(f"Loading model from {model_filename}")

    robot_model = load_robot_model(model_filename)
    adjacents = adjacent_collisions(robot_model)
    disable_collision(robot_model, adjacents, verbose=True)
    defaults = default_collisions(robot_model)
    disable_collision(robot_model, defaults, verbose=True)
    disable_collisions = adjacents + defaults

    disable_collision_tags = [
        "\t"
        + DISABLE_COLLISIONS_TAG.format(
            link1=link1,
            link2=link2,
            reason=reason,
        )
        for link1, link2, reason in disable_collisions
    ]

    LOGGER.info(
        SRDF.format(disable_collisions="\n".join(disable_collision_tags)),
    )

    LOGGER.info("Done!")


if __name__ == "__main__":
    main()
