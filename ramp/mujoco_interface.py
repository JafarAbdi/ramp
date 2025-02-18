"""Python interface to mujoco_simulator."""

import json
import logging
from pathlib import Path

import mujoco
import numpy as np
import zenoh

from ramp.exceptions import MissingJointError
from ramp.hardware_interface import HardwareInterface
from ramp.robot_model import RobotModel
from ramp.robot_state import RobotState
from mujoco_simulator_msgs.mujoco_simulator_pb2 import (
    ResetModelRequest,
    AddVisualGeometryRequest,
    RemoveVisualGeometryRequest,
    VisualGeometry,
    Pose,
)

LOGGER = logging.getLogger(__name__)
FILE_PATH = Path(__file__).parent
MOCAP_FILE_PATH = FILE_PATH / "mocap.xml"
MOCAP_CHILD_BODY_NAME = "target"
MOCAP_PREFIX = "mocap/target"


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

        self._session = zenoh.open(zenoh.Config())
        self._qpos_subscriber = self._session.declare_subscriber(
            "robot/qpos",
            zenoh.handlers.RingChannel(1),
        )
        self._qvel_subscriber = self._session.declare_subscriber(
            "robot/qvel",
            zenoh.handlers.RingChannel(1),
        )
        self._mocap_subscriber = self._session.declare_subscriber(
            "robot/mocap",
            zenoh.handlers.RingChannel(1),
        )
        self._ctrl_publisher = self._session.declare_publisher("robot/ctrl")

        self.reset(
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
            self.actuator_names[joint_name]: float(joint_position)
            for joint_name, joint_position in zip(
                joint_names,
                joint_positions,
                strict=True,
            )
        }

    def read(self) -> RobotState:
        """Read robot state."""
        qpos = np.asarray(self.qpos())
        # > qvel = self.qvel()
        # > qvel=self.from_mj_joint_positions(qvel),
        return RobotState.from_actuated_qpos(
            self.robot_model,
            self.from_mj_joint_positions(qpos),
        )

    def write(self, robot_state: RobotState):
        """Write control commands to robot."""
        self.ctrl(
            self.as_mj_ctrl(
                robot_state.robot_model.joint_names,
                robot_state.actuated_qpos(),
            ),
        )

    def add_decorative_geometry(
        self,
        name: str,
        geom_type: str,
        pos: list[float],
        size: list[float],
    ):
        """Add a decorative geometry object to the simulator.

        The object will be displayed in the GUI but will not interact with the simulation. If object with the same name it will be replaced.

        Args:
            name: The name of the geometry object.
            geom_type: The type of the geometry object (e.g., "box").
            pos: The position of the geometry object as [x, y, z].
            size: The size of the geometry object as [x, y, z].

        Raises:
            RuntimeError: If the add geometry object request fails.
        """
        assert len(pos) == 3
        assert len(size) == 3
        request = AddVisualGeometryRequest()
        request.name = name
        geometry = request.visual_geometry
        geometry.type = geom_type
        geometry.pose.pos.extend(pos)
        geometry.pose.quat.extend([1.0, 0.0, 0.0, 0.0])
        geometry.size.extend(size)
        geometry.rgba.extend([0.0, 1.0, 0.0, 1.0])
        replies = list(
            self._session.get(
                "add_geometry",
                payload=request.SerializeToString(),
            ),
        )
        assert len(replies) == 1

    def remove_decorative_geometry(self, name: str):
        """Remove a decorative geometry object from the simulator.

        Args:
            name: The name of the geometry object.

        Raises:
            RuntimeError: If the remove geometry object request fails.
        """
        request = RemoveVisualGeometryRequest()
        request.name = name
        replies = list(
            self._session.get(
                "remove_geometry",
                payload=request.SerializeToString(),
            ),
        )
        assert len(replies) == 1

    # TODO: Add removing models + Same for mocap
    def attach_model(  # noqa: PLR0913
        self,
        model_filename: str,
        parent_body_name: str,
        child_body_name: str,
        pose: tuple[list[float], list[float]],
        prefix: str,
        site_name: str | None = None,
    ):
        """Send a reset request to the simulator.

        Args:
            model_filename: The filename of the model to load.
            parent_body_name: The name of the parent body (e.g., "world").
            child_body_name: The name of the child body (e.g., "robot"), should exist in @model_filename.
            pose: The pose of the child body in the parent body frame as ([x, y, z], [qx, qy, qz, qw]).
            prefix: The prefix to use for the child body.
            site_name: The name of the site to attach the child

        Raises:
            RuntimeError: If the attach model request fails.
        """
        pos, quat = pose
        assert len(pos) == 3
        assert len(quat) == 4
        mj_quat = [
            quat[3],
            quat[0],
            quat[1],
            quat[2],
        ]  # mujoco uses [qw, qx, qy, qz] instead of [qx, qy, qz, qw]
        replies = list(
            self._session.get(
                "attach_model",
                payload=zenoh.ext.z_serialize(
                    json.dumps(
                        {
                            "model_filename": str(
                                Path(model_filename).resolve(),
                            ),
                            "parent_body_name": parent_body_name,
                            "child_body_name": child_body_name,
                            "site_name": site_name or "",
                            "pos": pos,
                            "quat": mj_quat,
                            "prefix": prefix,
                            "suffix": "",
                        },
                    ).encode(),
                ),
            ),
        )
        assert len(replies) == 1
        ok = zenoh.ext.z_deserialize(bool, replies[0].ok.payload)
        if not ok:
            msg = "Failed to attach model"
            raise RuntimeError(msg)

    def reset(self, *, model_filename: str | None = None, keyframe: str | None = None):
        """Send a reset request to the simulator.

        Args:
            model_filename: The filename of the model to load.
            keyframe: The name of the keyframe to use after loading/resetting the model.

        Raises:
            RuntimeError: If the reset request fails.
        """
        request = ResetModelRequest()
        if model_filename:
            request.model_filename = str(Path(model_filename).resolve())
        if keyframe:
            request.keyframe = keyframe
        attachments = {}
        replies = list(
            self._session.get(
                "reset",
                payload=zenoh.ZBytes(request.SerializeToString()),
            ),
        )
        assert len(replies) == 1
        ok, error_msg = zenoh.ext.z_deserialize(tuple[bool, str], replies[0].ok.payload)
        if not ok:
            msg = f"Failed to reset the simulation: {error_msg}"
            raise RuntimeError(msg)

    def get_model_filename(self):
        """Get the filename of the model loaded in the simulator.

        Returns:
            The filename of the model loaded in the simulator.

        Raises:
            RuntimeError: If the request fails.
        """
        replies = list(self._session.get("model"))
        assert len(replies) == 1
        try:
            return replies[0].ok.payload.to_string()
        except Exception:
            LOGGER.exception("Failed to get the model filename")
            return None

    def qpos(self):
        """Get the current qpos.

        Returns:
            The current qpos.
        """
        while (qpos := self._qpos_subscriber.try_recv()) is None:
            pass
        return zenoh.ext.z_deserialize(list[float], qpos.payload)

    def qvel(self):
        """Get the current qvel.

        Returns:
            The current qvel.
        """
        while (qvel := self._qvel_subscriber.try_recv()) is None:
            pass
        return zenoh.ext.z_deserialize(list[float], qvel.payload)

    def add_mocap(self, parent_body_name: str, pose: tuple[list[float], list[float]]):
        """Add a mocap to the simulator.

        Args:
            parent_body_name: The name of the parent body.
            pose: The pose of the mocap in the parent body frame as ([x, y, z], [qx, qx, qy, qw]).
        """
        self.attach_model(
            MOCAP_FILE_PATH,
            parent_body_name,
            MOCAP_CHILD_BODY_NAME,
            pose,
            MOCAP_PREFIX,
        )

    def mocap(self):
        """Get the current mocap.

        Returns:
            The current mocap as ([x, y, z], [qw, qx, qy, qz]).
        """
        while (mocap := self._mocap_subscriber.try_recv()) is None:
            pass
        mocap_pose = json.loads(zenoh.ext.z_deserialize(str, mocap.payload))
        return (
            mocap_pose["pos"],
            mocap_pose["quat"],
        )

    def ctrl(self, ctrl: dict[int, float]):
        """Send ctrl to the simulator.

        Args:
            ctrl: The control signal to send to the simulator.
        """
        self._ctrl_publisher.put(zenoh.ext.z_serialize(ctrl))
