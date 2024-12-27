"""A robot class that doesn't depend on ROS."""

import contextlib
import importlib
import io
import logging
import os
import re
import sys
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile

import casadi
import numpy as np
import pink
import pinocchio
import pinocchio.visualize
import toml
import xacrodoc
from pink import solve_ik
from pink.tasks import FrameTask
from pinocchio import casadi as cpin
from rich import pretty
from rich.logging import RichHandler
from urdf_parser_py import urdf as urdf_parser
from xacrodoc import XacroDoc

from ramp.constants import (
    MAX_ERROR,
    MAX_ITERATIONS,
    MUJOCO_DESCRIPTION_VARIANT,
    PINOCCHIO_PLANAR_JOINT,
    PINOCCHIO_UNBOUNDED_JOINT,
    ROBOT_DESCRIPTION_PREFIX,
)
from ramp.exceptions import (
    MissingAccelerationLimitError,
    MissingBaseLinkError,
    MissingJointError,
    RobotDescriptionNotFoundError,
)

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=os.getenv("LOG_LEVEL", "INFO").upper())


@contextlib.contextmanager
def filter_urdf_parser_stderr():
    """A context manager that filters stderr for urdf_parser_py's annoying errors."""
    filters = [r"Unknown tag", r"Unknown attribute"]
    original_stderr = sys.stderr
    string_io = io.StringIO()

    class FilteredStderr:
        def write(self, message):
            if not any(re.search(pattern, message) for pattern in filters):
                original_stderr.write(message)
            string_io.write(message)

        def flush(self):
            original_stderr.flush()

    sys.stderr = FilteredStderr()
    try:
        yield string_io
    finally:
        sys.stderr = original_stderr


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
    # Joint id != Joint indices, so we need a mapping between them
    joint_id_to_indices = {}
    for idx, joint in enumerate(model.joints):
        joint_id_to_indices[joint.id] = idx
    return joint_id_to_indices


def joint_ids_to_indices(model: pinocchio.Model) -> dict[int, list[int]]:
    """Return a mapping from joint ids to joint indices."""

    def joint_indices(joint: pinocchio.JointModel):
        """Return the joint indices."""
        joint_type = joint.shortname()
        if re.match(PINOCCHIO_UNBOUNDED_JOINT, joint_type):
            return [joint.idx_q]
        if (
            joint_type == PINOCCHIO_PLANAR_JOINT
        ):  # x, y, theta (Theta is continuous so 2 values, but we will handle it in the (as/from)_pinocchio_joint_position_continuous)
            return [joint.idx_q + idx for idx in range(3)]
        return [joint.idx_q + idx for idx in range(joint.nq)]

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


def load_mimic_joints(
    robot_description: Path,
    model: pinocchio.Model,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the mimic joints indices, multipliers, offsets and mimicked joint indices.

    Args:
        robot_description: The robot description file path
        model: The pinocchio model

    Returns:
        The mimic joint indices, multipliers, offsets and mimicked joint indices.
    """
    if robot_description.suffix != ".urdf":
        return np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])
    with filter_urdf_parser_stderr():
        urdf = urdf_parser.URDF.from_xml_file(robot_description)
    mimic_joint_indices = []
    mimic_joint_multipliers = []
    mimic_joint_offsets = []
    mimicked_joint_indices = []
    joint_id_to_indices = joint_ids_to_indices(model)
    for joint in urdf.joints:
        if joint.mimic is not None:
            mimicked_joint_index = joint_id_to_indices[
                model.getJointId(joint.mimic.joint)
            ]
            mimic_joint_index = joint_id_to_indices[model.getJointId(joint.name)]
            assert (
                len(mimic_joint_index) == 1
            ), f"Only single joint mimic supported {mimic_joint_index}"
            assert (
                len(mimicked_joint_index) == 1
            ), f"Only single mimicked joint is supported {mimicked_joint_index}"
            mimicked_joint_indices.append(mimicked_joint_index[0])
            mimic_joint_indices.append(mimic_joint_index[0])
            mimic_joint_multipliers.append(joint.mimic.multiplier or 1.0)
            mimic_joint_offsets.append(joint.mimic.offset or 0.0)
    return (
        np.asarray(mimic_joint_indices),
        np.asarray(mimic_joint_multipliers),
        np.asarray(mimic_joint_offsets),
        np.asarray(mimicked_joint_indices),
    )


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
    joint_acceleration_limits: InitVar[dict[str, float]]
    model: pinocchio.Model
    collision_model: pinocchio.GeometryModel
    visual_model: pinocchio.GeometryModel
    groups: dict[str, GroupModel]
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
            mimic_joint_indices,
            mimic_joint_multipliers,
            mimic_joint_offsets,
            mimicked_joint_indices,
        ) = load_mimic_joints(self.model_filename, self.model)
        object.__setattr__(self, "mimic_joint_indices", mimic_joint_indices)
        object.__setattr__(self, "mimic_joint_multipliers", mimic_joint_multipliers)
        object.__setattr__(self, "mimic_joint_offsets", mimic_joint_offsets)
        object.__setattr__(self, "mimicked_joint_indices", mimicked_joint_indices)
        joint_names = []
        for group in self.groups.values():
            joint_names.extend(group.joints)
        object.__setattr__(self, "joint_names", joint_names)

        acceleration_limits = []
        for joint_name in self.joint_names:
            if (
                joint_acceleration_limit := joint_acceleration_limits.get(joint_name)
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


def _get_robot_description_path(robot_description: str) -> Path:
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


def _load_xacro(
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

    return (Path(parsed_file_path), pinocchio.buildModelsFromUrdf(parsed_file_path))


def _load_models(
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
            return _load_xacro(robot_description_path, mappings)
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


def _make_groups(model: pinocchio.Model, configs):
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


class RobotState:
    """A class to represent the robot state."""

    def __init__(
        self,
        robot_model: RobotModel,
        qpos: np.ndarray | None = None,
        qvel: np.ndarray | None = None,
    ):
        """Initialize the robot state.

        Args:
            robot_model: The robot model to use for the state representation
            qpos: The joint positions. Initialized to the neutral position if None
            qvel: The joint velocities. Initialized to zeros if None
        """
        self.robot_model = robot_model
        self.qpos = pinocchio.neutral(robot_model.model) if qpos is None else qpos
        self.qvel = np.zeros(robot_model.model.nv) if qvel is None else qvel

    def __getitem__(self, key):
        """Get the actuated joint positions for a group."""
        qpos = get_converted_qpos(self.robot_model, self.qpos)
        return qpos[self.robot_model[key].joint_position_indices]

    def __setitem__(self, key, value):
        """Set the actuated joint positions for a group."""
        assert (
            len(value)
            == len(
                self.robot_model.groups[key].joint_position_indices,
            )
        ), f"Expected {len(self.robot_model[key].joints)} joint positions, got {len(value)}"
        self.qpos[self.robot_model[key].joint_position_indices] = value
        apply_pinocchio_mimic_joints(self.robot_model, self.qpos)
        apply_pinocchio_continuous_joints(self.robot_model, self.qpos)

    def clone(self):
        """Clone the robot state."""
        return RobotState(self.robot_model, np.copy(self.qpos), np.copy(self.qvel))

    @property
    def actuated_qpos(self) -> np.ndarray:
        """Return the actuated joint positions."""
        qpos = get_converted_qpos(self.robot_model, self.qpos)
        return np.concatenate(
            [
                qpos[group.joint_position_indices]
                for group in self.robot_model.groups.values()
            ],
        )

    @classmethod
    def from_actuated_qpos(cls, robot_model: RobotModel, joint_positions: np.ndarray):
        """Create a robot state from actuated joint positions."""
        qpos = pinocchio.neutral(robot_model.model)
        # Loop through the groups and set the joint positions
        start_index = 0
        for group in robot_model.groups.values():
            qpos[group.joint_position_indices] = joint_positions[
                start_index : start_index + len(group.joint_position_indices)
            ]
            start_index += len(group.joint_position_indices)
        apply_pinocchio_mimic_joints(robot_model, qpos)
        apply_pinocchio_continuous_joints(robot_model, qpos)
        return cls(robot_model, qpos)

    @classmethod
    def from_pinocchio_qpos(cls, robot_model: RobotModel, q: np.ndarray):
        """Convert pinocchio joint positions to joint positions."""
        assert len(q) == robot_model.model.nq
        return cls(robot_model, q)

    @classmethod
    def from_named_state(
        cls,
        robot_model: RobotModel,
        group_name: str,
        state_name: str,
    ):
        """Create a robot state from a named state."""
        return cls(
            robot_model,
            as_pinocchio_qpos(
                robot_model,
                pinocchio.neutral(robot_model.model),
                group_name,
                robot_model[group_name].named_states[state_name],
            ),
        )

    def get_frame_pose(
        self,
        target_frame_name,
    ) -> pinocchio.SE3:
        """Get the pose of a frame."""
        data = self.robot_model.model.createData()
        target_frame_id = self.robot_model.model.getFrameId(target_frame_name)
        pinocchio.framesForwardKinematics(
            self.robot_model.model,
            data,
            self.qpos,
        )
        try:
            return data.oMf[target_frame_id]
        except IndexError as index_error:
            raise pink.exceptions.FrameNotFound(
                target_frame_name,
                self.robot_model.model.frames,
            ) from index_error

    def jacobian(
        self,
        target_frame_name,
        reference_frame=pinocchio.ReferenceFrame.LOCAL,
    ):
        """Calculate the Jacobian of a frame.

        Args:
            target_frame_name: The target frame name
            reference_frame: The reference frame

        Returns:
            The Jacobian matrix of shape (6, n) where n is the number of joints.
        """
        data = self.robot_model.model.createData()
        return pinocchio.computeFrameJacobian(
            self.robot_model.model,
            data,
            self.qpos,
            self.robot_model.model.getFrameId(target_frame_name),
            reference_frame,
        )

    def differential_ik(
        self,
        group_name: str,
        target_pose,
        iteration_callback=None,
    ):
        """Compute the inverse kinematics of the robot for a given target pose.

        Args:
            group_name: The group name to compute the IK for
            target_pose: The target pose [x, y, z, qx, qy, qz, qw] or 4x4 homogeneous transformation matrix
            iteration_callback: Callback function after each iteration

        Returns:
            The joint positions for the target pose or None if no solution was found
        """
        assert self.robot_model[
            group_name
        ].tcp_link_name, f"tcp_link_name is not defined for group '{group_name}'"
        group = self.robot_model[group_name]

        end_effector_task = FrameTask(
            self.robot_model[group_name].tcp_link_name,
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=1.0,  # [cost] / [rad]
        )

        data: pinocchio.Data = self.robot_model.model.createData()
        end_effector_task.set_target(as_pinocchio_pose(target_pose))
        dt = 0.01
        configuration = pink.Configuration(
            self.robot_model.model,
            data,
            self.qpos,
        )
        number_of_iterations = 0
        actuated_joints_velocities = np.zeros(self.robot_model.model.nv)
        while number_of_iterations < MAX_ITERATIONS:
            # Compute velocity and integrate it into next configuration
            # Only update the actuated joints' velocities
            velocity = solve_ik(
                configuration,
                [end_effector_task],
                dt,
                solver="quadprog",
            )

            actuated_joints_velocities[group.joint_velocity_indices] = velocity[
                group.joint_velocity_indices
            ]
            configuration.integrate_inplace(actuated_joints_velocities, dt)

            if iteration_callback is not None:
                iteration_callback(RobotState(self.robot_model, configuration.q))
            if (
                np.linalg.norm(end_effector_task.compute_error(configuration))
                < MAX_ERROR
            ):
                qpos = get_converted_qpos(self.robot_model, configuration.q)
                self[group_name] = qpos[group.joint_position_indices]
                return True
            configuration = pink.Configuration(
                self.robot_model.model,
                data,
                configuration.q,
            )
            number_of_iterations += 1
        return False


class Robot:
    """Robot base class."""

    def __init__(
        self,
        config_path: Path,
    ) -> None:
        """Init.

        Args:
            config_path: Path to the config file configs.toml
        """
        if not config_path.exists():
            msg = f"Config file does not exist: {config_path}"
            raise FileNotFoundError(msg)
        configs = toml.load(config_path)

        mappings = configs["robot"].get("mappings", {})
        model_filename, (model, collision_model, visual_model) = _load_models(
            config_path.parent
            / _get_robot_description_path(
                configs["robot"]["description"],
            ),
            mappings=mappings,
        )

        collision_model.addAllCollisionPairs()
        self._geometry_objects = {}
        verbose = LOGGER.level == logging.DEBUG
        if srdf_path := configs["robot"].get("disable_collisions"):
            pinocchio.removeCollisionPairsFromXML(
                model,
                collision_model,
                load_xacro(
                    config_path.parent / srdf_path,
                    mappings=mappings,
                ),
                verbose=verbose,
            )
        if verbose:
            pretty.pprint(configs)

        link_names = [frame.name for frame in model.frames]
        self.base_link = configs["robot"]["base_link"]
        if self.base_link not in link_names:
            msg = f"Base link '{self.base_link}' not in link names: {link_names}"
            raise MissingBaseLinkError(
                msg,
            )

        self.robot_model = RobotModel(
            model_filename,
            configs["acceleration_limits"],
            model,
            collision_model,
            visual_model,
            _make_groups(model, configs),
        )

    def add_object(self, name: str, geometry_object, pose: pinocchio.SE3):
        """Add an object to the robot's collision model.

        Args:
            name: Name of the object
            geometry_object: Geometry object
            pose: Pose of the object
        """
        geometry_object.name = name
        geometry_object.parentJoint = 0
        geometry_object.meshColor = np.array([1.0, 0.0, 0.0, 1.0])
        geometry_object.placement = pose
        geometry_object_collision_id = (
            self.robot_model.collision_model.addGeometryObject(
                geometry_object,
            )
        )
        self._geometry_objects[name] = geometry_object_collision_id
        self.robot_model.visual_model.addGeometryObject(geometry_object)
        for geometry_id in range(
            len(self.robot_model.collision_model.geometryObjects)
            - len(self._geometry_objects),
        ):
            self.robot_model.collision_model.addCollisionPair(
                pinocchio.CollisionPair(self._geometry_objects[name], geometry_id),
            )


# TODO: Maybe delete and combine with Robot class?
# Prefix with c for CasADi
# Example: cmodel, cdata, cq, cjacobian
class CasADiRobot:
    """A class to represent the robot in CasADi."""

    def __init__(self, robot: Robot):
        """Initialize the CasADi robot.

        Args:
            robot: The robot to convert to CasADi.
        """
        self.model = cpin.Model(robot.robot_model.model)
        self.data = self.model.createData()
        self.q = casadi.SX.sym("q", robot.robot_model.model.nq, 1)
        cpin.framesForwardKinematics(self.model, self.data, self.q)
        cpin.updateFramePlacements(self.model, self.data)

    def jacobian(
        self,
        target_frame_name,
        reference_frame=pinocchio.ReferenceFrame.LOCAL,
    ):
        """Calculate the Jacobian of a frame.

        Args:
            target_frame_name: The target frame name
            reference_frame: The reference frame

        Returns:
            The Jacobian matrix of shape (6, n) where n is the number of joints.
        """
        return cpin.computeFrameJacobian(
            self.model,
            self.data,
            self.q,
            self.model.getFrameId(target_frame_name),
            reference_frame,
        )


def print_collision_results(
    collision_model: pinocchio.GeometryModel,
    collision_results,
):
    """Print the collision results.

    Args:
        collision_model: The collision model
        collision_results: The collision results
    """
    for k in range(len(collision_model.collisionPairs)):
        cr = collision_results[k]
        cp = collision_model.collisionPairs[k]
        if cr.isCollision():
            LOGGER.debug(
                f"Collision between: ({collision_model.geometryObjects[cp.first].name}"
                f", {collision_model.geometryObjects[cp.second].name})",
            )


def check_collision(robot_state: RobotState, *, verbose=False):
    """Check if the robot is in collision with the given joint positions.

    Args:
        robot_state: The robot state to check for collision
        verbose: Whether to print the collision results.

    Returns:
        True if the robot is in collision, False otherwise.
    """
    robot_model = robot_state.robot_model
    data = robot_model.model.createData()
    collision_data = robot_model.collision_model.createData()

    pinocchio.computeCollisions(
        robot_model.model,
        data,
        robot_model.collision_model,
        collision_data,
        robot_state.qpos,
        stop_at_first_collision=not verbose,
    )
    if verbose:
        print_collision_results(
            robot_model.collision_model,
            collision_data.collisionResults,
        )
    return np.any([cr.isCollision() for cr in collision_data.collisionResults])
