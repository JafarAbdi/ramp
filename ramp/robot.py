"""A robot class that doesn't depend on ROS."""

import contextlib
import importlib
import io
import logging
import os
import re
import sys
from dataclasses import dataclass, field, InitVar
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
    GROUP_NAME,
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


def joint_ids_to_indices(model: pinocchio.Model) -> dict[int, int]:
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


def as_pinocchio_joint_position_continuous(q, joint_index: int | np.ndarray):
    """Convert continuous joint positions to pinocchio joint positions."""
    q[joint_index], q[joint_index + 1] = (
        np.cos(q[joint_index]),
        np.sin(q[joint_index]),
    )


def from_pinocchio_joint_positions_continuous(
    q,
    joint_index: int | np.ndarray,
):
    """Convert pinocchio joint positions to continuous joint positions."""
    q[joint_index] = np.arctan2(q[joint_index + 1], q[joint_index])


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
    joints: list[str]
    joint_indices: np.ndarray
    joint_position_indices: np.ndarray
    joint_velocity_indices: np.ndarray
    named_states: dict[str, np.ndarray] = field(default_factory=dict)
    tcp_link_name: str | None = None


@dataclass(frozen=True, slots=True)
class RobotModel:
    model_filename: InitVar[Path]
    model: pinocchio.Model
    collision_model: pinocchio.GeometryModel
    visual_model: pinocchio.GeometryModel
    groups: dict[str, GroupModel]
    joint_names: list[str] = field(init=False)
    mimic_joint_indices: np.ndarray = field(init=False)
    mimic_joint_multipliers: np.ndarray = field(init=False)
    mimic_joint_offsets: np.ndarray = field(init=False)
    mimicked_joint_indices: np.ndarray = field(init=False)
    continuous_joint_indices: np.ndarray = field(init=False)

    def __post_init__(self, model_filename):
        object.__setattr__(
            self, "continuous_joint_indices", get_continuous_joint_indices(self.model)
        )
        (
            mimic_joint_indices,
            mimic_joint_multipliers,
            mimic_joint_offsets,
            mimicked_joint_indices,
        ) = load_mimic_joints(model_filename, self.model)
        object.__setattr__(self, "mimic_joint_indices", mimic_joint_indices)
        object.__setattr__(self, "mimic_joint_multipliers", mimic_joint_multipliers)
        object.__setattr__(self, "mimic_joint_offsets", mimic_joint_offsets)
        object.__setattr__(self, "mimicked_joint_indices", mimicked_joint_indices)
        joint_names = []
        for group in self.groups.values():
            joint_names.extend(group.joints)
        object.__setattr__(self, "joint_names", joint_names)

    def __getitem__(self, key):
        return self.groups[key]

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
            ]
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
            ]
        )


@dataclass(slots=True)
class GroupState:
    name: str
    qpos: np.ndarray


class Robot:
    """Robot base class."""

    def __init__(  # noqa: C901
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
        model, collision_model, visual_model = self._load_models(
            config_path.parent
            / self._get_robot_description_path(
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

        self.link_names = [frame.name for frame in model.frames]
        self.base_link = configs["robot"]["base_link"]
        if self.base_link not in self.link_names:
            msg = f"Base link '{self.base_link}' not in link names: {self.link_names}"
            raise MissingBaseLinkError(
                msg,
            )

        groups = self._make_groups(model, configs)

        joint_names = []
        for group in groups.values():
            joint_names.extend(group.joints)

        acceleration_limits = configs["acceleration_limits"]
        self.acceleration_limits = []
        for joint_name in joint_names:
            if (
                joint_acceleration_limit := acceleration_limits.get(joint_name)
            ) is None:
                raise MissingAccelerationLimitError(
                    joint_names,
                    acceleration_limits.keys(),
                )
            self.acceleration_limits.append(joint_acceleration_limit)
        self.acceleration_limits = np.asarray(self.acceleration_limits)

        self.robot_model = RobotModel(
            self.model_filename,
            model,
            collision_model,
            visual_model,
            groups,
        )

    def _get_robot_description_path(self, robot_description: str) -> Path:
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
        self, robot_description_path: Path, mappings: dict
    ) -> tuple[pinocchio.Model, pinocchio.GeometryModel, pinocchio.GeometryModel]:
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
        self.model_filename = Path(parsed_file_path)

        return pinocchio.buildModelsFromUrdf(parsed_file_path)

    def _load_models(
        self, robot_description_path: Path, mappings: dict
    ) -> tuple[pinocchio.Model, pinocchio.GeometryModel, pinocchio.GeometryModel]:
        """Load the model/collision & visual models.

        Args:
            robot_description_path: Path to the robot description file
            mappings: Mappings for the xacro file
        """
        match robot_description_path.suffix:
            case ".xacro" | ".urdf":
                return self._load_xacro(robot_description_path, mappings)
            case ".xml":
                self.model_filename = robot_description_path
                return pinocchio.shortcuts.buildModelsFromMJCF(
                    str(self.model_filename),
                    verbose=True,
                )
            case _:
                msg = f"Unknown robot description file type: {robot_description_path}"
                raise ValueError(msg)

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

    def _make_groups(self, model: pinocchio.Model, configs):
        groups = {}
        joint_id_to_indices = joint_ids_to_indices(model)
        joint_id_to_velocity_indices = joint_ids_to_velocity_indices(model)
        joint_id_to_name_indices = joint_ids_to_name_indices(model)
        for group_name, group_config in configs["group"].items():
            joints = group_config["joints"]
            tcp_link_name = group_config.get("tcp_link_name")
            if tcp_link_name is not None:
                assert (
                    tcp_link_name in self.link_names
                ), f"Group {group_name} TCP link '{tcp_link_name}' not in link names: {self.link_names}"
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
                joint = model.joints[joint_id]
                actuated_joint_indices.append(joint_id_to_name_indices[joint_id])
                actuated_joint_position_indices.extend(joint_id_to_indices[joint_id])
                actuated_joint_velocity_indices.append(
                    joint_id_to_velocity_indices[joint_id],
                )
            named_states = {}
            for state_name, state_config in group_config.get(
                "named_states", {}
            ).items():
                assert len(state_config) == len(
                    actuated_joint_position_indices,
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

    def get_frame_pose(self, joint_positions, target_frame_name) -> pinocchio.SE3:
        """Get the pose of a frame."""
        data = self.model.createData()
        target_frame_id = self.model.getFrameId(target_frame_name)
        pinocchio.framesForwardKinematics(
            self.model,
            data,
            self.as_pinocchio_joint_positions(joint_positions),
        )
        try:
            return data.oMf[target_frame_id]
        except IndexError as index_error:
            raise pink.exceptions.FrameNotFound(
                target_frame_name,
                self.model.frames,
            ) from index_error

    def jacobian(
        self,
        joint_positions,
        target_frame_name,
        reference_frame=pinocchio.ReferenceFrame.LOCAL,
    ):
        """Calculate the Jacobian of a frame.

        Args:
            joint_positions: The joint positions
            target_frame_name: The target frame name
            reference_frame: The reference frame

        Returns:
            The Jacobian matrix of shape (6, n) where n is the number of joints.
        """
        data = self.model.createData()
        return pinocchio.computeFrameJacobian(
            self.model,
            data,
            self.as_pinocchio_joint_positions(joint_positions),
            self.model.getFrameId(target_frame_name),
            reference_frame,
        )

    def differential_ik(
        self,
        target_pose,
        initial_configuration=None,
        iteration_callback=None,
    ):
        """Compute the inverse kinematics of the robot for a given target pose.

        Args:
            target_pose: The target pose [x, y, z, qx, qy, qz, qw] or 4x4 homogeneous transformation matrix
            initial_configuration: The initial configuration
            iteration_callback: Callback function after each iteration

        Returns:
            The joint positions for the target pose or None if no solution was found
        """
        assert len(initial_configuration) == len(self.joint_names)
        if iteration_callback is None:

            def iteration_callback(_):
                return None

        assert self.groups[
            GROUP_NAME
        ].tcp_link_name, f"tcp_link_name is not defined for group '{GROUP_NAME}'"

        end_effector_task = FrameTask(
            self.groups[GROUP_NAME].tcp_link_name,
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=1.0,  # [cost] / [rad]
        )

        data: pinocchio.Data = self.model.createData()
        end_effector_task.set_target(as_pinocchio_pose(target_pose))
        dt = 0.01
        configuration = pink.Configuration(
            self.model,
            data,
            self.as_pinocchio_joint_positions(initial_configuration),
        )
        number_of_iterations = 0
        actuated_joints_velocities = np.zeros(self.model.nv)
        while number_of_iterations < MAX_ITERATIONS:
            # Compute velocity and integrate it into next configuration
            # Only update the actuated joints' velocities
            velocity = solve_ik(
                configuration,
                [end_effector_task],
                dt,
                solver="quadprog",
            )

            actuated_joints_velocities[self.actuated_joint_velocity_indices] = velocity[
                self.actuated_joint_velocity_indices
            ]
            configuration.integrate_inplace(actuated_joints_velocities, dt)

            iteration_callback(self.from_pinocchio_joint_positions(configuration.q))
            if (
                np.linalg.norm(end_effector_task.compute_error(configuration))
                < MAX_ERROR
            ):
                return self.from_pinocchio_joint_positions(configuration.q)
            configuration = pink.Configuration(self.model, data, configuration.q)
            number_of_iterations += 1
        return None

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
        for actuated_joint_index in self.actuated_joint_indices:
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
        self.model = cpin.Model(robot.model)
        self.data = self.model.createData()
        self.q = casadi.SX.sym("q", robot.model.nq, 1)
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


class RobotState:

    def __init__(
        self, robot_model: RobotModel, qpos0: np.ndarray | GroupState | None = None
    ):
        self.robot_model = robot_model
        self.qpos = pinocchio.neutral(robot_model.model)
        if qpos0 is not None:
            self.qpos = self.as_pinocchio_joint_positions(qpos0)

    @staticmethod
    def from_group_state(robot_model: RobotModel, group_state: GroupState):
        """Create a robot state from a group state."""
        robot_state = RobotState(robot_model)
        robot_state.qpos[
            robot_model.groups[group_state.name].joint_position_indices
        ] = group_state.qpos
        return robot_state

    # TODO: Should this return GroupState?
    def __getitem__(self, key):
        return self.qpos[self.robot_model.groups[key].joint_position_indices]

    def clone(self, group_state: GroupState | None = None):
        """Clone the robot state.

        Args:
            group_state: The group state to set.

        Returns:
            The cloned robot state with the group state set.
        """
        qpos = self.qpos.copy()
        if group_state is not None:
            assert (
                group_state.name in self.robot_model.groups
            ), f"Unknown group: {group_state.name} in {self.robot_model.groups.keys()}"
            assert len(group_state.qpos) == len(
                self.robot_model.groups[group_state.name].joint_position_indices,
            ), f"Expected {len(self.robot_model.groups[group_state.name].joint_position_indices)} joint positions, got {len(group_state.qpos)}"
            qpos[self.robot_model.groups[group_state.name].joint_position_indices] = (
                group_state.qpos
            )
        return RobotState(self.robot_model, qpos)

    # TODO: Update docs
    # TODO: Maybe rename to as_pinocchio_qpos?
    # Convert actuated qpos or group actuated qpos to pinocchio qpos
    def as_pinocchio_joint_positions(
        self, joint_positions: np.ndarray | GroupState
    ) -> np.ndarray:
        """Convert joint positions to pinocchio joint positions."""
        q = self.qpos.copy()
        if isinstance(joint_positions, GroupState):
            q[self.groups[joint_positions.name].joint_position_indices] = (
                joint_positions.qpos
            )
        else:
            # Loop through the groups and set the joint positions
            start_index = 0
            for group in self.robot_model.groups.values():
                q[group.joint_position_indices] = joint_positions[
                    start_index : start_index + len(group.joint_position_indices)
                ]
                start_index += len(group.joint_position_indices)

        if self.robot_model.mimic_joint_indices.size != 0:
            q[self.robot_model.mimic_joint_indices] = (
                q[self.robot_model.mimicked_joint_indices]
                * self.robot_model.mimic_joint_multipliers
                + self.robot_model.mimic_joint_offsets
            )
        if self.robot_model.continuous_joint_indices.size != 0:
            as_pinocchio_joint_position_continuous(
                q,
                self.robot_model.continuous_joint_indices,
            )
        return q

    def actuated_qpos(self) -> np.ndarray:
        """Return the actuated joint positions."""
        qpos = self.qpos.copy()
        if self.robot_model.continuous_joint_indices.size != 0:
            from_pinocchio_joint_positions_continuous(
                qpos,
                self.robot_model.continuous_joint_indices,
            )
        return np.concatenate(
            [
                qpos[group.joint_position_indices]
                for group in self.robot_model.groups.values()
            ]
        )

    # # TODO: Rename to numpy()? Or active qpos? Actuated qpos?
    # def from_pinocchio_joint_positions(self, q: np.ndarray) -> np.ndarray:
    #     """Convert pinocchio joint positions to joint positions."""
    #     assert len(q) == self.robot_model.model.nq
    #     joint_positions = np.copy(q)
    #     if self.robot_model.continuous_joint_indices.size != 0:
    #         from_pinocchio_joint_positions_continuous(
    #             joint_positions,
    #             self.robot_model.continuous_joint_indices,
    #         )
    #     return np.concatenate(
    #         [
    #             joint_positions[group.joint_position_indices]
    #             for group in self.robot_model.groups.values()
    #         ]
    #     )


def print_collision_results(
    collision_model: pinocchio.GeometryModel, collision_results
):
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
        joint_positions: Joint positions of the robot.
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
            robot_model.collision_model, collision_data.collisionResults
        )
    return np.any([cr.isCollision() for cr in collision_data.collisionResults])
