"""A robot class that doesn't depend on ROS."""

import logging
import os
from dataclasses import InitVar, dataclass, field
from pathlib import Path

import hppfcl
import numpy as np
import pinocchio
import pinocchio.visualize
import toml
from rich import pretty

from ramp.constants import ROBOT_DESCRIPTION_PREFIX
from ramp.exceptions import (
    MissingAccelerationLimitError,
    MissingBaseLinkError,
    MissingJointError,
)
from ramp.pinocchio_utils import (
    get_continuous_joint_indices,
    get_robot_description_path,
    joint_ids_to_indices,
    joint_ids_to_name_indices,
    joint_ids_to_velocity_indices,
    load_mimic_joints,
    load_models,
    load_xacro,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GroupModel:
    """A class to represent a group of joints."""

    joints: list[str]
    joint_indices: np.ndarray
    joint_position_indices: np.ndarray
    joint_velocity_indices: np.ndarray
    named_states: dict[str, np.ndarray] = field(default_factory=dict)
    tcp_link_name: str | None = None


@dataclass(frozen=True, slots=True)
class RobotModel:
    """A class to represent a robot model."""

    model_filename: Path
    base_link: str
    model: pinocchio.Model
    collision_model: pinocchio.GeometryModel
    visual_model: pinocchio.GeometryModel
    groups: dict[str, GroupModel]
    motion_model: dict
    joint_acceleration_limits: InitVar[dict[str, float]]
    acceleration_limits: np.ndarray = field(init=False)
    joint_names: list[str] = field(init=False)
    mimic_joint_indices: np.ndarray = field(init=False)
    mimic_joint_multipliers: np.ndarray = field(init=False)
    mimic_joint_offsets: np.ndarray = field(init=False)
    mimicked_joint_indices: np.ndarray = field(init=False)
    continuous_joint_indices: np.ndarray = field(init=False)

    def __post_init__(self, joint_acceleration_limits):
        """Initialize the robot model."""
        object.__setattr__(
            self,
            "continuous_joint_indices",
            get_continuous_joint_indices(self.model),
        )
        (
            mimic_joint_ids,
            mimic_joint_indices,
            mimic_joint_multipliers,
            mimic_joint_offsets,
            mimicked_joint_indices,
        ) = load_mimic_joints(self.model_filename, self.model)
        # Check if any of the joints is a mimic joint
        for group in self.groups.values():
            for joint_name in group.joints:
                joint_id = self.model.getJointId(joint_name)
                if joint_id in mimic_joint_ids:
                    msg = f"Joint '{joint_name}' is a mimic joint, which is not allowed in groups."
                    raise ValueError(
                        msg,
                    )
        object.__setattr__(self, "mimic_joint_indices", mimic_joint_indices)
        object.__setattr__(self, "mimic_joint_multipliers", mimic_joint_multipliers)
        object.__setattr__(self, "mimic_joint_offsets", mimic_joint_offsets)
        object.__setattr__(self, "mimicked_joint_indices", mimicked_joint_indices)
        joint_names = []
        for group in self.groups.values():
            joint_names.extend(group.joints)
        object.__setattr__(self, "joint_names", joint_names)

        acceleration_limits = []
        if joint_acceleration_limits:
            for joint_name in self.joint_names:
                if (
                    joint_acceleration_limit := joint_acceleration_limits.get(
                        joint_name,
                    )
                ) is None:
                    raise MissingAccelerationLimitError(
                        self.joint_names,
                        joint_acceleration_limits.keys(),
                    )
                acceleration_limits.append(joint_acceleration_limit)
        object.__setattr__(self, "acceleration_limits", np.asarray(acceleration_limits))

    def __getitem__(self, key):
        """Get a group by name."""
        return self.groups[key]

    @property
    def position_limits(self) -> list[tuple[float, float]]:
        """Return the joint limits.

        Returns:
            List of tuples of (lower, upper) joint limits.
        """
        # TODO: How to handle continuous joints?
        # https://github.com/stack-of-tasks/pinocchio/issues/794
        # https://github.com/stack-of-tasks/pinocchio/issues/777
        joint_limits = []
        for group in self.groups.values():
            for actuated_joint_index in group.joint_position_indices:
                joint = self.model.joints[int(actuated_joint_index)]
                joint_limits.append(
                    (
                        self.model.lowerPositionLimit[
                            actuated_joint_index : actuated_joint_index + joint.nq
                        ],
                        self.model.upperPositionLimit[
                            actuated_joint_index : actuated_joint_index + joint.nq
                        ],
                    ),
                )
        return joint_limits

    @property
    def velocity_limits(self) -> list[float]:
        """Return the velocity limits.

        Returns:
            List of velocity limits.
        """
        return np.concatenate(
            [
                self.model.velocityLimit[group.joint_position_indices]
                for group in self.groups.values()
            ],
        )

    @property
    def effort_limits(self) -> list[float]:
        """Return the effort limits.

        Returns:
            List of effort limits.
        """
        return np.concatenate(
            [
                self.model.effortLimit[group.joint_position_indices]
                for group in self.groups.values()
            ],
        )

    def body_geometries(self, name: str) -> list[pinocchio.GeometryObject]:
        """Get the body geometries for a link.

        Args:
            name: The link name

        Returns:
            The body geometries
        """
        return filter(
            lambda geometry: geometry.parentFrame == self.model.getBodyId(name),
            self.collision_model.geometryObjects,
        )


def make_groups(model: pinocchio.Model, configs: dict) -> dict[str, GroupModel]:
    """Make groups from the model and the config.

    Args:
        model: The pinocchio model
        configs: Configs for the groups

    Returns:
        A dictionary of group models
    """
    groups = {}
    joint_id_to_indices = joint_ids_to_indices(model)
    joint_id_to_velocity_indices = joint_ids_to_velocity_indices(model)
    joint_id_to_name_indices = joint_ids_to_name_indices(model)
    for group_name, group_config in configs["group"].items():
        joints = group_config["joints"]
        tcp_link_name = group_config.get("tcp_link_name")
        link_names = [frame.name for frame in model.frames]
        if tcp_link_name is not None:
            assert (
                tcp_link_name in link_names
            ), f"Group {group_name} TCP link '{tcp_link_name}' not in link names: {link_names}"
        for joint_name in joints:
            if joint_name not in model.names:
                msg = f"Joint '{joint_name}' for group '{group_name}' not in model joints: {list(model.names)}"
                raise MissingJointError(
                    msg,
                )
        # TODO: Extend the comment about joint indices != joint ids
        actuated_joint_indices = []
        actuated_joint_position_indices = []
        actuated_joint_velocity_indices = []
        for joint_name in joints:
            joint_id = model.getJointId(joint_name)
            actuated_joint_indices.append(joint_id_to_name_indices[joint_id])
            actuated_joint_position_indices.extend(joint_id_to_indices[joint_id])
            actuated_joint_velocity_indices.append(
                joint_id_to_velocity_indices[joint_id],
            )
        named_states = {}
        for state_name, state_config in group_config.get(
            "named_states",
            {},
        ).items():
            assert (
                len(state_config)
                == len(
                    actuated_joint_position_indices,
                )
            ), f"Named state '{state_name}' has {len(state_config)} joint positions, expected {len(actuated_joint_position_indices)} for {joints}"
            named_states[state_name] = np.asarray(state_config)
        groups[group_name] = GroupModel(
            joints=joints,
            joint_indices=np.asarray(actuated_joint_indices),
            joint_position_indices=np.asarray(actuated_joint_position_indices),
            joint_velocity_indices=np.asarray(actuated_joint_velocity_indices),
            named_states=named_states,
            tcp_link_name=tcp_link_name,
        )
    return groups


def load_robot_model(
    config_path: Path,
    motion_models: dict | None = None,
    acceleration_limits: dict | None = None,
) -> RobotModel:
    """Load the robot model from a config file, URDF, XACRO, or MJCF's XML file.

    Args:
        config_path: Path to the config file, URDF, XACRO, or MJCF's XML file
        motion_models: Motion models for the robot (only used for URDF, XACRO, or MJCF's XML files)
        acceleration_limits: Acceleration limits for the robot (only used for URDF, XACRO, or MJCF's XML files)

    Returns:
        The robot model
    """
    if str(config_path).startswith(ROBOT_DESCRIPTION_PREFIX):
        # If the config path starts with the ROBOT_DESCRIPTION_PREFIX, we assume it's a robot description
        # and load the robot description from the path.
        config_path = get_robot_description_path(config_path)
    if not config_path.exists():
        msg = f"File does not exist: {config_path}"
        raise FileNotFoundError(msg)
    match config_path.suffix:
        case ".toml":
            configs = toml.load(config_path)
            configs["robot"]["description"] = str(
                config_path.parent
                / get_robot_description_path(configs["robot"]["description"]),
            )
            if srdf_path := configs["robot"].get("disable_collisions"):
                configs["robot"]["disable_collisions"] = str(
                    config_path.parent / srdf_path,
                )
        case ".urdf" | ".xacro" | ".xml":
            model_filename, (model, _, _) = load_models(
                config_path,
                {},
            )
            joint_names = []
            for joint in model.names:
                if joint == "universe":
                    continue
                joint_names.append(joint)
            configs = {
                "robot": {
                    "description": (
                        str(model_filename)
                        if model_filename.suffix == ".xacro"
                        else str(config_path)
                    ),
                    "base_link": "universe",
                },
                "group": {
                    "default": {
                        "joints": joint_names,
                    },
                },
                "acceleration_limits": acceleration_limits or {},
                "motion_model": motion_models or {},
            }
    return load_robot_model_from_configs(configs)


def load_robot_model_from_configs(configs: dict) -> RobotModel:
    """Load the robot model from the configs.

    Args:
        configs: Configs for the robot model

    Returns:
        The robot model
    """
    mappings = configs["robot"].get("mappings", {})
    model_filename, (model, collision_model, visual_model) = load_models(
        Path(configs["robot"]["description"]),
        mappings=mappings,
    )

    collision_model.addAllCollisionPairs()
    verbose = os.environ.get("LOG_LEVEL", "INFO").upper() == "DEBUG"
    if srdf_path := configs["robot"].get("disable_collisions"):
        pinocchio.removeCollisionPairsFromXML(
            model,
            collision_model,
            load_xacro(
                Path(srdf_path),
                mappings=mappings,
            ),
            verbose=verbose,
        )
    if verbose:
        pretty.pprint(configs)

    link_names = [frame.name for frame in model.frames]
    base_link = configs["robot"]["base_link"]
    if base_link not in link_names:
        msg = f"Base link '{base_link}' not in link names: {link_names}"
        raise MissingBaseLinkError(
            msg,
        )

    return RobotModel(
        model_filename,
        base_link,
        model,
        collision_model,
        visual_model,
        make_groups(model, configs),
        configs.get("motion_model", {}),
        configs.get("acceleration_limits", {}),
    )


def as_pinocchio_qpos(
    robot_model: RobotModel,
    reference_qpos: np.ndarray,
    group_name: str,
    group_qpos: np.ndarray,
) -> np.ndarray:
    """Convert joint positions to pinocchio joint positions."""
    q = np.copy(reference_qpos)
    q[robot_model[group_name].joint_position_indices] = group_qpos

    apply_pinocchio_mimic_joints(robot_model, q)
    apply_pinocchio_continuous_joints(robot_model, q)
    return q


def from_pinocchio_qpos_continuous(
    robot_model: RobotModel,
    q,
):
    """Convert pinocchio joint positions to continuous joint positions."""
    if robot_model.continuous_joint_indices.size == 0:
        return
    q[robot_model.continuous_joint_indices] = np.arctan2(
        q[robot_model.continuous_joint_indices + 1],
        q[robot_model.continuous_joint_indices],
    )


def get_converted_qpos(robot_model: RobotModel, qpos: np.ndarray) -> np.ndarray:
    """Get a copy of joint positions converted from Pinocchio's internal representation.

    Args:
        robot_model: The robot model to use for the conversion
        qpos: The joint positions to convert

    Returns:
        Converted joint positions with continuous joint handling
    """
    qpos = np.copy(qpos)
    from_pinocchio_qpos_continuous(robot_model, qpos)
    return qpos


def apply_pinocchio_continuous_joints(robot_model: RobotModel, q: np.ndarray):
    """Convert continuous joint positions to pinocchio joint positions."""
    if robot_model.continuous_joint_indices.size == 0:
        return
    (
        q[robot_model.continuous_joint_indices],
        q[robot_model.continuous_joint_indices + 1],
    ) = (
        np.cos(q[robot_model.continuous_joint_indices]),
        np.sin(q[robot_model.continuous_joint_indices]),
    )


def apply_pinocchio_mimic_joints(robot_model: RobotModel, q: np.ndarray):
    """Apply mimic joints to the joint positions."""
    if robot_model.mimic_joint_indices.size == 0:
        return
    q[robot_model.mimic_joint_indices] = (
        q[robot_model.mimicked_joint_indices] * robot_model.mimic_joint_multipliers
        + robot_model.mimic_joint_offsets
    )


# > hppfcl.Capsule(radius, height)
# > mesh_loader = hppfcl.MeshLoader()
# > mesh_loader.load(filename)
# > hppfcl.Cylinder(radius, height)
# > hppfcl.Box([x, y, z])
# > hppfcl.Sphere(radius)
# > hppfcl.Ellipsoid([x, y, z])


def create_geometry_object(
    name: str,
    geometry: hppfcl.CollisionGeometry,
    pose: pinocchio.SE3,
):
    """Create a geometry object.

    Args:
        name: The name of the geometry object
        geometry: hppfcl collision geometry
        pose: The pose of the geometry object

    Returns:
        The geometry object
    """
    geometry_object = pinocchio.GeometryObject(name, 0, pose, geometry)
    geometry_object.meshColor = np.array([0.0, 1.0, 0.0, 1.0])
    return geometry_object
