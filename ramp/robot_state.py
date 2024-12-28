"""A module to represent the robot state."""

import logging

import numpy as np
import pink
import pinocchio
import pinocchio.visualize
from pink import solve_ik
from pink.tasks import FrameTask

from ramp.constants import (
    MAX_ERROR,
    MAX_ITERATIONS,
)
from ramp.pinocchio_utils import (
    as_pinocchio_pose,
)
from ramp.robot_model import (
    RobotModel,
    apply_pinocchio_continuous_joints,
    apply_pinocchio_mimic_joints,
    as_pinocchio_qpos,
    get_converted_qpos,
)

LOGGER = logging.getLogger(__name__)


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
        assert qpos is None or len(qpos) == robot_model.model.nq
        self.robot_model = robot_model
        self.qpos = pinocchio.neutral(robot_model.model) if qpos is None else qpos
        self.qvel = np.zeros(robot_model.model.nv) if qvel is None else qvel
        self.geometry_objects = {}

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
        robot_state = RobotState(
            self.robot_model,
            np.copy(self.qpos),
            np.copy(self.qvel),
        )
        robot_state.geometry_objects = self.geometry_objects.copy()
        return robot_state

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

    # TODO: This will modify the robot_model, we need a better way to handle objects
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
        self.geometry_objects[name] = geometry_object_collision_id
        self.robot_model.visual_model.addGeometryObject(geometry_object)
        for geometry_id in range(
            len(self.robot_model.collision_model.geometryObjects)
            - len(self.geometry_objects),
        ):
            self.robot_model.collision_model.addCollisionPair(
                pinocchio.CollisionPair(self.geometry_objects[name], geometry_id),
            )

    def check_collision(self, *, verbose=False):
        """Check if the robot is in collision with the given joint positions.

        Args:
            robot_state: The robot state to check for collision
            verbose: Whether to print the collision results.

        Returns:
            True if the robot is in collision, False otherwise.
        """
        data = self.robot_model.model.createData()
        collision_data = self.robot_model.collision_model.createData()

        pinocchio.computeCollisions(
            self.robot_model.model,
            data,
            self.robot_model.collision_model,
            collision_data,
            self.qpos,
            stop_at_first_collision=not verbose,
        )
        if verbose:
            print_collision_results(
                self.robot_model.collision_model,
                collision_data.collisionResults,
            )
        return np.any([cr.isCollision() for cr in collision_data.collisionResults])