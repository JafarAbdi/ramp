"""A robot class that doesn't depend on ROS."""

import contextlib
import importlib
import io
import logging
import os
import re
import sys
from dataclasses import dataclass
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


def filter_values_by_joint_names(
    keys: list[str],
    values: list[float],
    joint_names: list[str],
) -> list[float]:
    """Filter values by joint names."""
    filtered_values = []
    for joint_name in joint_names:
        try:
            index = keys.index(joint_name)
        except ValueError:
            msg = f"Joint name '{joint_name}' not in input keys {keys}"
            raise ValueError(msg) from None
        filtered_values.append(values[index])
    return filtered_values


@dataclass(slots=True)
class Gripper:
    """Gripper configs.

    Args:
        open_value: Open position
        close_value: Close position
        actuated_joint: Name of the actuated joint
    """

    open_value: float
    close_value: float
    actuated_joint: str


@dataclass(slots=True)
class Group:
    """Group configs.

    Args:
        joints: List of joint names
        tcp_link_name (optional): Name of the TCP link
        gripper (optional): Gripper configs
    """

    joints: list[str]
    tcp_link_name: str | None = None
    gripper: Gripper | None = None


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

        self.mimic_joint_indices = np.asarray([])
        self.mimic_joint_multipliers = np.asarray([])
        self.mimic_joint_offsets = np.asarray([])
        self.mimicked_joint_indices = np.asarray([])

        mappings = configs["robot"].get("mappings", {})
        self._load_models(
            config_path.parent
            / self._get_robot_description_path(
                configs["robot"]["description"],
            ),
            mappings=mappings,
        )

        self.collision_model.addAllCollisionPairs()
        self._geometry_objects = {}
        verbose = LOGGER.level == logging.DEBUG
        if srdf_path := configs["robot"].get("disable_collisions"):
            pinocchio.removeCollisionPairsFromXML(
                self.model,
                self.collision_model,
                load_xacro(
                    config_path.parent / srdf_path,
                    mappings=mappings,
                ),
                verbose=verbose,
            )
        if verbose:
            pretty.pprint(configs)

        self.link_names = [frame.name for frame in self.model.frames]
        self.base_link = configs["robot"]["base_link"]
        if self.base_link not in self.link_names:
            msg = f"Base link '{self.base_link}' not in link names: {self.link_names}"
            raise MissingBaseLinkError(
                msg,
            )

        self.groups = self._make_groups(configs)

        self.joint_names = []
        for group in self.groups.values():
            self.joint_names.extend(group.joints)
            if group.gripper is not None:
                self.joint_names.append(group.gripper.actuated_joint)

        acceleration_limits = configs["acceleration_limits"]
        self.acceleration_limits = []
        for joint_name in self.joint_names:
            if (
                joint_acceleration_limit := acceleration_limits.get(joint_name)
            ) is None:
                raise MissingAccelerationLimitError(
                    self.joint_names,
                    acceleration_limits.keys(),
                )
            self.acceleration_limits.append(joint_acceleration_limit)
        self.acceleration_limits = np.asarray(self.acceleration_limits)

        # TODO: Should this be group specific????
        # TODO: Extend the comment about joint indices != joint ids
        joint_id_to_indices = self.joint_ids_to_indices()
        joint_id_to_velocity_indices = self.joint_ids_to_velocity_indices()
        joint_id_to_name_indices = self.joint_ids_to_name_indices()
        actuated_joint_indices = []
        actuated_joint_position_indices = []
        actuated_joint_velocity_indices = []
        for joint_name in self.joint_names:
            joint_id = self.model.getJointId(joint_name)
            joint = self.model.joints[joint_id]
            actuated_joint_indices.append(joint_id_to_name_indices[joint_id])
            actuated_joint_position_indices.extend(joint_id_to_indices[joint_id])
            actuated_joint_velocity_indices.append(
                joint_id_to_velocity_indices[joint_id],
            )
        self.actuated_joint_indices = np.asarray(actuated_joint_indices)
        self.actuated_joint_position_indices = np.asarray(
            actuated_joint_position_indices,
        )
        self.actuated_joint_velocity_indices = np.asarray(
            actuated_joint_velocity_indices,
        )
        # Continuous joints have 2 q, it's represented as cos(theta), sin(theta)
        continuous_joint_indices = []
        for joint in self.model.joints:
            joint_type = joint.shortname()
            if re.match(PINOCCHIO_UNBOUNDED_JOINT, joint_type):
                continuous_joint_indices.append(joint.idx_q)
            elif joint_type == PINOCCHIO_PLANAR_JOINT:
                continuous_joint_indices.append(
                    joint.idx_q + 2,
                )  # theta of the planar joint is continuous
        self.continuous_joint_indices = np.asarray(continuous_joint_indices)

        self.named_states = self._make_named_states(configs)

        self._ik_solver = None

    def joint_ids_to_name_indices(self) -> dict[int, int]:
        """Return a mapping from joint ids to joint indices."""
        # Joint id != Joint indices, so we need a mapping between them
        joint_id_to_indices = {}
        for idx, joint in enumerate(self.model.joints):
            joint_id_to_indices[joint.id] = idx
        return joint_id_to_indices

    def joint_ids_to_indices(self) -> dict[int, int]:
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
        for joint in self.model.joints:
            joint_id_to_indices[joint.id] = joint_indices(joint)
        return joint_id_to_indices

    def joint_ids_to_velocity_indices(self) -> dict[int, int]:
        """Return a mapping from joint ids to joint velocity indices."""
        # Joint id != Joint indices, so we need a mapping between them
        joint_id_to_indices = {}
        for joint in self.model.joints:
            joint_id_to_indices[joint.id] = joint.idx_v
        return joint_id_to_indices

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

    def _load_xacro(self, robot_description_path: Path, mappings: dict) -> Path:
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
        with filter_urdf_parser_stderr():
            urdf = urdf_parser.URDF.from_xml_string(robot_description)
        # Loading Pinocchio model
        with NamedTemporaryFile(
            mode="w",
            prefix="pinocchio_model_",
            suffix=".urdf",
            delete=False,
        ) as parsed_file:
            parsed_file_path = parsed_file.name
            parsed_file.write(robot_description)
        models: tuple[pinocchio.Model, pinocchio.Model, pinocchio.Model] = (
            pinocchio.buildModelsFromUrdf(parsed_file_path)
        )
        (
            self.model,
            self.collision_model,
            self.visual_model,
        ) = models

        mimic_joint_indices = []
        mimic_joint_multipliers = []
        mimic_joint_offsets = []
        mimicked_joint_indices = []
        joint_id_to_indices = self.joint_ids_to_indices()
        for joint in urdf.joints:
            if joint.mimic is not None:
                mimicked_joint_index = joint_id_to_indices[
                    self.model.getJointId(joint.mimic.joint)
                ]
                mimic_joint_index = joint_id_to_indices[
                    self.model.getJointId(joint.name)
                ]
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
        self.mimic_joint_indices = np.asarray(mimic_joint_indices)
        self.mimic_joint_multipliers = np.asarray(mimic_joint_multipliers)
        self.mimic_joint_offsets = np.asarray(mimic_joint_offsets)
        self.mimicked_joint_indices = np.asarray(mimicked_joint_indices)
        return Path(parsed_file_path)

    def _load_models(self, robot_description_path: Path, mappings: dict) -> None:
        """Load the model/collision & visual models.

        Args:
            robot_description_path: Path to the robot description file
            mappings: Mappings for the xacro file
        """
        match robot_description_path.suffix:
            case ".xacro" | ".urdf":
                self.model_filename = self._load_xacro(robot_description_path, mappings)
            case ".xml":
                self.model_filename = robot_description_path
                models: tuple[pinocchio.Model, pinocchio.Model, pinocchio.Model] = (
                    pinocchio.shortcuts.buildModelsFromMJCF(
                        str(self.model_filename),
                        verbose=True,
                    )
                )
                self.model, self.collision_model, self.visual_model = models
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
        geometry_object_collision_id = self.collision_model.addGeometryObject(
            geometry_object,
        )
        self._geometry_objects[name] = geometry_object_collision_id
        self.visual_model.addGeometryObject(geometry_object)
        for geometry_id in range(
            len(self.collision_model.geometryObjects) - len(self._geometry_objects),
        ):
            self.collision_model.addCollisionPair(
                pinocchio.CollisionPair(self._geometry_objects[name], geometry_id),
            )

    @staticmethod
    def _make_gripper_from_configs(gripper_configs):
        if gripper_configs is None:
            return None
        return Gripper(
            open_value=gripper_configs["open"],
            close_value=gripper_configs["close"],
            actuated_joint=gripper_configs["actuated_joint"],
        )

    def _make_groups(self, configs):
        groups = {}
        for group_name, group_config in configs["group"].items():
            groups[group_name] = Group(
                joints=group_config["joints"],
                tcp_link_name=group_config.get("tcp_link_name"),
                gripper=self._make_gripper_from_configs(group_config.get("gripper")),
            )
            if groups[group_name].tcp_link_name is not None:
                assert (
                    groups[group_name].tcp_link_name in self.link_names
                ), f"Group {group_name} TCP link '{groups[group_name].tcp_link_name}' not in link names: {self.link_names}"
            if groups[group_name].gripper is not None:
                assert (
                    groups[group_name].gripper.actuated_joint in self.model.names
                ), f"Gripper's actuated joint '{groups[group_name].gripper.actuated_joint}' not in model joints: {list(self.model.names)}"
            for joint_name in groups[group_name].joints:
                if joint_name not in self.model.names:
                    msg = f"Joint '{joint_name}' for group '{group_name}' not in model joints: {list(self.model.names)}"
                    raise MissingJointError(
                        msg,
                    )
        return groups

    def _make_named_states(self, configs):
        named_states = {}
        for state_name, state_config in configs["named_states"].items():
            assert (
                len(state_config)
                == len(
                    self.actuated_joint_position_indices,
                )
            ), f"Named state '{state_name}' has {len(state_config)} joint positions, expected {len(self.actuated_joint_position_indices)} for {self.joint_names}"
            named_states[state_name] = state_config

        return named_states

    def _print_collision_results(self, collision_results):
        for k in range(len(self.collision_model.collisionPairs)):
            cr = collision_results[k]
            cp = self.collision_model.collisionPairs[k]
            if cr.isCollision():
                LOGGER.debug(
                    f"Collision between: ({self.collision_model.geometryObjects[cp.first].name}"
                    f", {self.collision_model.geometryObjects[cp.second].name})",
                )

    def check_collision(self, joint_positions, *, verbose=False):
        """Check if the robot is in collision with the given joint positions.

        Args:
            joint_positions: Joint positions of the robot.
            verbose: Whether to print the collision results.

        Returns:
            True if the robot is in collision, False otherwise.
        """
        data = self.model.createData()
        collision_data = self.collision_model.createData()

        pinocchio.computeCollisions(
            self.model,
            data,
            self.collision_model,
            collision_data,
            self.as_pinocchio_joint_positions(joint_positions),
            stop_at_first_collision=not verbose,
        )
        if verbose:
            self._print_collision_results(collision_data.collisionResults)
        return np.any([cr.isCollision() for cr in collision_data.collisionResults])

    def _from_pinocchio_joint_positions_continuous(
        self,
        q,
        joint_index: int | np.ndarray,
    ):
        """Convert pinocchio joint positions to continuous joint positions."""
        q[joint_index] = np.arctan2(q[joint_index + 1], q[joint_index])

    def from_pinocchio_joint_positions(self, q):
        """Convert pinocchio joint positions to joint positions."""
        joint_positions = np.copy(q)
        if self.continuous_joint_indices.size != 0:
            self._from_pinocchio_joint_positions_continuous(
                joint_positions,
                self.continuous_joint_indices,
            )
        return joint_positions[self.actuated_joint_position_indices]

    def _as_pinocchio_joint_position_continuous(self, q, joint_index: int | np.ndarray):
        """Convert continuous joint positions to pinocchio joint positions."""
        q[joint_index], q[joint_index + 1] = (
            np.cos(q[joint_index]),
            np.sin(q[joint_index]),
        )

    def as_pinocchio_joint_positions(self, joint_positions):
        """Convert joint positions to pinocchio joint positions."""
        assert len(joint_positions) == len(
            self.actuated_joint_position_indices,
        ), f"{len(joint_positions)} != {len(self.actuated_joint_position_indices)}"
        q = pinocchio.neutral(self.model)
        q[self.actuated_joint_position_indices] = joint_positions
        if self.mimic_joint_indices.size != 0:
            q[self.mimic_joint_indices] = (
                q[self.mimicked_joint_indices] * self.mimic_joint_multipliers
                + self.mimic_joint_offsets
            )
        if self.continuous_joint_indices.size != 0:
            self._as_pinocchio_joint_position_continuous(
                q,
                self.continuous_joint_indices,
            )
        return q

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

    @property
    def velocity_limits(self) -> list[float]:
        """Return the velocity limits.

        Returns:
            List of velocity limits.
        """
        return self.model.velocityLimit[self.actuated_joint_position_indices]

    @property
    def effort_limits(self) -> list[float]:
        """Return the effort limits.

        Returns:
            List of effort limits.
        """
        return self.model.effortLimit[self.actuated_joint_position_indices]


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
