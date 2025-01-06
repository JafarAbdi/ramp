"""Hardware interface for reading and writing robot state."""

from abc import ABC, abstractmethod

import mujoco
import numpy as np
import zenoh
from mujoco_simulator_py.mujoco_interface import MuJoCoInterface

from ramp.exceptions import MissingJointError
from ramp.robot_model import RobotModel
from ramp.robot_state import RobotState


class HardwareInterface(ABC):
    """Abstract class for hardware interface."""

    def __init__(self, robot_model: RobotModel):
        """Initialize hardware interface."""
        self.robot_model = robot_model

    @abstractmethod
    def read(self) -> RobotState:
        """Read robot state."""

    @abstractmethod
    def write(self, joint_names: list[str], ctrl: np.ndarray):
        """Write control commands to robot."""


class MockHardwareInterface(HardwareInterface):
    """Mock hardware interface for testing."""

    def __init__(self, robot_model: RobotModel, initial_robot_state: RobotState):
        """Initialize mock hardware interface.

        Args:
            robot_model: Robot model.
            initial_robot_state: Initial robot state.
        """
        super().__init__(robot_model)
        self._robot_state = initial_robot_state.clone()

    def read(self) -> RobotState:
        """Read robot state."""
        return self._robot_state.clone()

    def write(self, joint_names: list[str], ctrl: np.ndarray):
        """Write control commands to robot."""
        msg = "Mock hardware interface does not support writing."
        raise NotImplementedError(msg)


class MuJoCoHardwareInterface(HardwareInterface):
    """MuJoCo hardware interface."""

    def __init__(self, robot_model: RobotModel, keyframe: str | None = None):
        """Initialize MuJoCo hardware interface.

        Args:
            robot_model: Robot model.
            keyframe: Keyframe for the mujoco simulation.
        """
        super().__init__(robot_model)
        # Current assumption: The model file is in the same directory as the scene file
        # ROBOT.xml -> Used for pinocchio since it doesn't support builtin textures
        # or having a worldbody without a body tag
        # <worldbody>
        #   <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
        #   <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
        # </worldbody>
        # scene.xml -> Used for mujoco simulator
        assert robot_model.model_filename.suffix == ".xml"
        self.model_filename = self.robot_model.model_filename.parent / "scene.xml"
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
        for group_name, group in self.robot_model.groups.items():
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
            self.group_actuated_joint_indices[group_name] = np.asarray(
                actuated_joint_indices,
            )

            zenoh.init_log_from_env_or("error")
            self._mj_interface = MuJoCoInterface()
            self._mj_interface.reset(
                model_filename=self.model_filename,
                keyframe=keyframe,
            )

    def from_mj_joint_positions(self, q) -> np.ndarray:
        """Convert mujoco joint positions to joint positions."""
        return np.concatenate(
            [
                q[self.group_actuated_joint_indices[group_name]]
                for group_name in self.robot_model.groups
            ],
        )

    def as_mj_joint_positions(self, joint_positions):
        """Convert joint positions to mujoco joint positions."""
        q = self.model.qpos0.copy()
        start_index = 0
        for group_name, group in self.robot_model.groups.items():
            # TODO: Multi-dof joints
            q[self.group_actuated_joint_indices[group_name]] = joint_positions[
                start_index : start_index + len(group.joints)
            ]
            start_index += len(group.joints)
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

    def read(self) -> RobotState:
        """Read robot state."""
        qpos = np.asarray(self._mj_interface.qpos())
        # > qvel = self._mj_interface.qvel()
        # > qvel=self.from_mj_joint_positions(qvel),
        return RobotState.from_actuated_qpos(
            self.robot_model,
            self.from_mj_joint_positions(qpos),
        )

    def write(self, joint_names: list[str], ctrl: np.ndarray):
        """Write control commands to robot."""
        self._mj_interface.ctrl(
            self.as_mj_ctrl(
                joint_names,
                ctrl,
            ),
        )
