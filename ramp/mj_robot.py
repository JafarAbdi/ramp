"""Mujoco robot class."""

import logging
import os
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from ramp.exceptions import MissingJointError

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=os.getenv("LOG_LEVEL", "INFO").upper())

GROUP_NAME = "arm"


# Inherit from mujoco interface??
class MjRobot:
    """Mujoco robot class."""

    def __init__(self, model_filename: Path, groups: dict):
        """Initialize the Mujoco robot.

        Args:
            model_filename: The Mujoco model filename.
            groups: The groups of the robot.
        """
        # Current assumption: The model file is in the same directory as the scene file
        # ROBOT.xml -> Used for pinocchio since it doesn't support builtin textures
        # or having a worldbody without a body tag
        # <worldbody>
        #   <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
        #   <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
        # </worldbody>
        # scene.xml -> Used for mujoco simulator
        self.model_filename = model_filename.parent / "scene.xml"
        self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(
            str(self.model_filename),
        )
        self.data: mujoco.MjData = mujoco.MjData(self.model)

        mimic_joint_indices = []
        mimic_joint_multipliers = []
        mimic_joint_offsets = []
        mimicked_joint_indices = []

        for obj1id, obj2id, equality_type, equality_solref in zip(
            self.model.eq_obj1id,
            self.model.eq_obj2id,
            self.model.eq_type,
            self.model.eq_solref,
            strict=False,
        ):
            # TODO: ID == INDEX???
            if mujoco.mjtEq(equality_type) == mujoco.mjtEq.mjEQ_JOINT:
                mimicked_joint_indices.append(self.model.joint(obj1id).qposadr[0])
                mimic_joint_indices.append(self.model.joint(obj2id).qposadr[0])
                mimic_joint_multipliers.append(
                    equality_solref[1],
                )  # TODO: Make index a variable
                mimic_joint_offsets.append(
                    equality_solref[0],
                )  # TODO: Make index a variable
        self.mimic_joint_indices = np.asarray(mimic_joint_indices)
        self.mimic_joint_multipliers = np.asarray(mimic_joint_multipliers)
        self.mimic_joint_offsets = np.asarray(mimic_joint_offsets)
        self.mimicked_joint_indices = np.asarray(mimicked_joint_indices)

        # joint name -> actuator id
        self.actuator_names = {}
        for i in range(self.model.nu):
            actuator = self.model.actuator(i)
            transmission_id, _ = actuator.trnid
            # TODO: Handle other transmission types
            if mujoco.mjtTrn(actuator.trntype) == mujoco.mjtTrn.mjTRN_JOINT:
                self.actuator_names[self.model.joint(transmission_id).name] = (
                    actuator.id
                )

        self.link_names = [self.model.body(i).name for i in range(self.model.nbody)]
        # Group -> Joint indices
        self.group_actuated_joint_indices = {}
        for group_name, group in groups.items():
            actuated_joint_indices = []
            for joint_name in group.joints:
                try:
                    joint = self.model.joint(joint_name)
                except KeyError as e:
                    msg = f"Joint '{joint_name}' not in model joints. {e}"
                    raise MissingJointError(msg) from e
                actuated_joint_indices.append(
                    joint.qposadr[0],
                )  # TODO: How about multi-dof joints?
            # TODO: Handle gripper actuated joint
            self.group_actuated_joint_indices[group_name] = np.asarray(
                actuated_joint_indices,
            )

        self.joint_names = []
        for group in groups.values():
            self.joint_names.extend(group.joints)
            if group.gripper is not None:
                self.joint_names.append(group.gripper.actuated_joint)

    def check_collision(self, joint_positions, *, verbose=False):
        """Check if the robot is in collision with the given joint positions.

        Args:
            joint_positions: Joint positions of the robot.
            verbose: Whether to print the collision results.

        Returns:
            True if the robot is in collision, False otherwise.
        """
        data = mujoco.MjData(self.model)
        data.qpos = self.as_mj_joint_positions(joint_positions)
        mujoco.mj_forward(self.model, data)
        if verbose:
            contacts = Counter()
            for contact in data.contact:
                body1_id = self.model.geom_bodyid[contact.geom1]
                body1_name = mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body1_id,
                )
                body2_id = self.model.geom_bodyid[contact.geom2]
                body2_name = mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body2_id,
                )
                contacts[(body1_name, body2_name)] += 1
            LOGGER.debug(f"Contacts: {contacts}")
        return data.ncon > 0

    def from_mj_joint_positions(self, q):
        """Convert mujoco joint positions to joint positions."""
        joint_positions = np.copy(q)
        return joint_positions[self.group_actuated_joint_indices[GROUP_NAME]]

    def as_mj_joint_positions(self, joint_positions):
        """Convert joint positions to mujoco joint positions."""
        q = self.model.qpos0.copy()
        q[self.group_actuated_joint_indices[GROUP_NAME]] = joint_positions
        if self.mimic_joint_indices.size != 0:
            q[self.mimic_joint_indices] = (
                q[self.mimicked_joint_indices] * self.mimic_joint_multipliers
                + self.mimic_joint_offsets
            )
        return q

    def as_mj_ctrl(self, joint_names, joint_positions):
        """Convert joint positions to mujoco ctrl."""
        return {
            self.actuator_names[joint_name]: joint_position
            for joint_name, joint_position in zip(
                joint_names,
                joint_positions,
                strict=True,
            )
        }

    def get_frame_pose(self, joint_positions, target_frame_name):
        """Get the pose of a frame."""
        data: mujoco.MjData = mujoco.MjData(self.model)
        data.qpos = joint_positions
        mujoco.mj_forward(self.model, data)
        target_frame_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            target_frame_name,
        )
        transform = np.eye(4)
        transform[:3, :3] = data.xmat[target_frame_id].reshape(3, 3)
        transform[:3, 3] = data.xpos[target_frame_id]
        return transform

    @property
    def position_limits(self) -> list[tuple[float, float]]:
        """Return the joint limits.

        Returns:
            List of tuples of (lower, upper) joint limits.
        """
        # TODO: Handle continuous joints (limited = 0)
        return self.model.jnt_range[self.group_actuated_joint_indices[GROUP_NAME]]

    @property
    def effort_limits(self) -> list[float]:
        """Return the effort limits.

        Returns:
            List of effort limits.
        """
        return self.model.effortLimit[self.group_actuated_joint_indices[GROUP_NAME]]
