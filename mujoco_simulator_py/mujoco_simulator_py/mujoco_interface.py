"""Python interface to mujoco_simulator."""

import json
import logging
import os
from pathlib import Path

import zenoh
from rich.logging import RichHandler

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=os.getenv("LOG_LEVEL", "INFO").upper())

FILE_PATH = Path(__file__).parent
MOCAP_FILE_PATH = FILE_PATH / "mocap.xml"
MOCAP_CHILD_BODY_NAME = "target"
MOCAP_PREFIX = "mocap/target"


class MuJoCoInterface:
    """Python interface to mujoco_simulator."""

    def __init__(self):
        """Initialize the interface."""
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
        replies = list(
            self._session.get(
                "add_geometry",
                payload=zenoh.ext.z_serialize(
                    (
                        name,
                        json.dumps(
                            {
                                "type": geom_type,
                                "pos": pos,
                                "size": size,
                            },
                        ).encode(),
                    ),
                ),
            ),
        )
        assert len(replies) == 1
        ok = zenoh.ext.z_deserialize(bool, replies[0].ok.payload)
        if not ok:
            msg = "Failed to add geometry object"
            raise RuntimeError(msg)

    def remove_decorative_geometry(self, name: str):
        """Remove a decorative geometry object from the simulator.

        Args:
            name: The name of the geometry object.

        Raises:
            RuntimeError: If the remove geometry object request fails.
        """
        replies = list(
            self._session.get(
                "remove_geometry",
                payload=zenoh.ext.z_serialize(name),
            ),
        )
        assert len(replies) == 1
        ok = zenoh.ext.z_deserialize(bool, replies[0].ok.payload)
        if not ok:
            msg = "Failed to add geometry object"
            raise RuntimeError(msg)

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
        attachments = {}
        if model_filename:
            attachments["model_filename"] = str(Path(model_filename).resolve())
        if keyframe:
            attachments["keyframe"] = keyframe
        replies = list(
            self._session.get(
                "reset",
                payload=zenoh.ext.z_serialize(obj=True),
                attachment=zenoh.ext.z_serialize(attachments),
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
