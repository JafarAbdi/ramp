"""Utility functions for working with Pinocchio models."""

import importlib
import logging
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pinocchio
import pinocchio.visualize
import xacrodoc
from xacrodoc import XacroDoc

from ramp.constants import (
    MUJOCO_DESCRIPTION_VARIANT,
    PINOCCHIO_PLANAR_JOINT,
    PINOCCHIO_UNBOUNDED_JOINT,
    ROBOT_DESCRIPTION_PREFIX,
)
from ramp.exceptions import (
    RobotDescriptionNotFoundError,
)


def as_pinocchio_pose(pose):
    """Convert pose to pinocchio pose."""
    pose = np.asarray(pose)
    if pose.shape == (7,):
        return pinocchio.XYZQUATToSE3(pose)
    if pose.shape == (4, 4):
        return pinocchio.SE3(pose)
    msg = f"Unknown target pose shape: {pose.shape} - it should be (7,) or (4, 4)"
    raise RuntimeError(
        msg,
    )


def get_continuous_joint_indices(model: pinocchio.Model) -> np.ndarray:
    """Get the continuous joint indices."""
    # Continuous joints have 2 q, it's represented as cos(theta), sin(theta)
    continuous_joint_indices = []
    for joint in model.joints:
        joint_type = joint.shortname()
        if re.match(PINOCCHIO_UNBOUNDED_JOINT, joint_type):
            continuous_joint_indices.append(joint.idx_q)
        elif joint_type == PINOCCHIO_PLANAR_JOINT:
            continuous_joint_indices.append(
                joint.idx_q + 2,
            )  # theta of the planar joint is continuous
    return np.asarray(continuous_joint_indices)


def joint_ids_to_name_indices(model: pinocchio.Model) -> dict[int, int]:
    """Return a mapping from joint ids to joint indices."""
    # Joint id == Joint indices except for the universe joint, so we need a mapping between them
    joint_id_to_indices = {}
    for idx, joint in enumerate(model.joints):
        joint_id_to_indices[joint.id] = idx
    return joint_id_to_indices


# TODO: Rename please
def joint_nq(joint: pinocchio.JointModel) -> int:
    """Return the number of joint positions."""
    joint_type = joint.shortname()
    # Continuous joints have 2 q, it's represented as cos(theta), sin(theta)
    if re.match(PINOCCHIO_UNBOUNDED_JOINT, joint_type):
        return 1
    if (
        joint_type == PINOCCHIO_PLANAR_JOINT
    ):  # x, y, theta (Theta is continuous so 2 values, but we will handle it in the (as/from)_pinocchio_joint_position_continuous)
        return 3
    return joint.nq


def joint_indices(joint: pinocchio.JointModel):
    """Return the joint indices."""
    return [joint.idx_q + idx for idx in range(joint_nq(joint))]


def joint_ids_to_indices(model: pinocchio.Model) -> dict[int, list[int]]:
    """Return a mapping from joint ids to joint indices."""
    # Joint id != Joint indices, so we need a mapping between them
    joint_id_to_indices = {}
    for joint in model.joints:
        joint_id_to_indices[joint.id] = joint_indices(joint)
    return joint_id_to_indices


def joint_ids_to_velocity_indices(model: pinocchio.Model) -> dict[int, int]:
    """Return a mapping from joint ids to joint velocity indices."""
    # Joint id != Joint indices, so we need a mapping between them
    joint_id_to_indices = {}
    for joint in model.joints:
        joint_id_to_indices[joint.id] = joint.idx_v
    return joint_id_to_indices


def get_robot_description_path(robot_description: str) -> Path:
    """Get the robot description path.

    Args:
        robot_description: The robot description path or name

    Returns:
        The robot description path
    """
    # Suppress git logs
    logging.getLogger("git").setLevel(logging.INFO)
    if robot_description.startswith(ROBOT_DESCRIPTION_PREFIX):
        robot_description_name = robot_description[len(ROBOT_DESCRIPTION_PREFIX) :]
        try:
            robot_description_module = importlib.import_module(
                f"robot_descriptions.{robot_description_name}",
            )
        except (ImportError, AttributeError) as import_error:
            raise RobotDescriptionNotFoundError(
                robot_description_name,
            ) from import_error
        if MUJOCO_DESCRIPTION_VARIANT in robot_description_name:
            return Path(robot_description_module.MJCF_PATH)
        return Path(robot_description_module.URDF_PATH)
    return Path(robot_description)


def load_models_from_xacro(
    robot_description_path: Path,
    mappings: dict,
) -> tuple[
    Path,
    tuple[pinocchio.Model, pinocchio.GeometryModel, pinocchio.GeometryModel],
]:
    """Load the model/collision & visual models from an xacro file.

    Args:
        robot_description_path: Path to the robot description file
        mappings: Mappings for the xacro file
    Returns:
        The robot description string from the xacro file
    """
    robot_description = load_xacro(
        robot_description_path,
        mappings,
    )
    # Loading Pinocchio model
    with NamedTemporaryFile(
        mode="w",
        prefix="pinocchio_model_",
        suffix=".urdf",
        delete=False,
    ) as parsed_file:
        parsed_file_path = parsed_file.name
        parsed_file.write(robot_description)

    return (
        Path(parsed_file_path),
        pinocchio.buildModelsFromUrdf(
            parsed_file_path,
            package_dirs=[robot_description_path.parent.resolve()],
            mimic=True,
        ),
    )


def load_models(
    robot_description_path: Path,
    mappings: dict,
) -> tuple[
    Path,
    tuple[pinocchio.Model, pinocchio.GeometryModel, pinocchio.GeometryModel],
]:
    """Load the model/collision & visual models.

    Args:
        robot_description_path: Path to the robot description file
        mappings: Mappings for the xacro file
    """
    match robot_description_path.suffix:
        case ".xacro" | ".urdf":
            return load_models_from_xacro(robot_description_path, mappings)
        case ".xml":
            return (
                robot_description_path,
                pinocchio.shortcuts.buildModelsFromMJCF(
                    str(robot_description_path),
                    verbose=True,
                ),
            )
        case _:
            msg = f"Unknown robot description file type: {robot_description_path}"
            raise ValueError(msg)


def load_xacro(file_path: Path, mappings: dict | None = None) -> str:
    """Load a xacro file and render it with the given mappings."""
    if not file_path.exists():
        msg = f"File {file_path} doesn't exist"
        raise FileNotFoundError(msg)

    if (conda_prefix := os.environ.get("CONDA_PREFIX")) is not None:
        # TODO: Should we automatically add all folders in $CONDA_PREFIX/share/..??
        xacrodoc.packages.update_package_cache(
            {
                "ur_description": f"{conda_prefix}/share/ur_description",
                "franka_description": f"{conda_prefix}/share/franka_description",
                "robotiq_description": f"{conda_prefix}/share/robotiq_description",
                "ur_robot_driver": f"{conda_prefix}/share/ur_robot_driver",
                "realsense2_description": f"{conda_prefix}/share/realsense2_description",
                "kortex_description": f"{conda_prefix}/share/kortex_description",
            },
        )
    return XacroDoc.from_file(
        file_path,
        subargs=mappings,
        resolve_packages=True,
    ).to_urdf_string()
