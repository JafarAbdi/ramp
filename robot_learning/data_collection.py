"""Data collection script for the SO100 robot."""

import logging
import os
import pathlib
import signal
import sys
from dataclasses import asdict
from pprint import pformat

import cv2
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.cameras.configs import IntelRealSenseCameraConfig
from lerobot.common.robot_devices.control_configs import (
    RecordControlConfig,
)
from lerobot.common.robot_devices.control_utils import (
    record_episode,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.configs import parser
from rich.logging import RichHandler

from robot_learning.keyboard_listener import KeyboardListener
from robot_learning.utils import (
    move_robots_to_initial_position,
    move_robots_to_safe_position,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)

LOGGER = logging.getLogger(__name__)
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()


def safe_reset(func):
    """Safe reset decorator to ensure the robot is in a safe state after an exception."""

    def wrapper(robot, *args, **kwargs):
        try:
            return func(robot, *args, **kwargs)
        except Exception as e:
            move_robots_to_safe_position(robot)
            if robot.is_connected:
                robot.disconnect()
            raise e from e

    return wrapper


# Based on lerobot/lerobot/common/robot_devices/control_utils.py
@safe_reset
def record(  # noqa: C901
    robot: Robot,
    cfg: RecordControlConfig,
) -> LeRobotDataset:
    """A function to record data from the robot and save it to a dataset.

    Args:
        robot: The robot to record data from.
        cfg: The configuration for the recording.
    """
    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.repo_id,
            root=cfg.root,
        )
        if len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.num_image_writer_processes,
                num_threads=cfg.num_image_writer_threads_per_camera
                * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.fps, cfg.video)
    else:
        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(cfg.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.repo_id,
            cfg.fps,
            root=cfg.root,
            robot=robot,
            use_videos=cfg.video,
            image_writer_processes=cfg.num_image_writer_processes,
            image_writer_threads=cfg.num_image_writer_threads_per_camera
            * len(robot.cameras),
        )

    # Load pretrained policy
    policy = (
        None
        if cfg.policy is None
        else make_policy(cfg.policy, cfg.device, ds_meta=dataset.meta)
    )

    if not robot.is_connected:
        robot.connect()

    keyboard_listener = KeyboardListener()

    recorded_episodes = 0
    recording = False
    while True:
        if keyboard_listener.start_recording() and not recording:
            keyboard_listener.events.dict_events["gripper"] = 100  # Open
            recording = True
            LOGGER.info("Reset the environment")
            move_robots_to_initial_position(robot)
            LOGGER.info(f"Recording episode {recorded_episodes}")
            robot.leader_arms["main"].write("Torque_Enable", 0)
            record_episode(
                robot=robot,
                dataset=dataset,
                events=keyboard_listener.events.dict_events,
                episode_time_s=float("inf"),
                display_cameras=cfg.display_cameras,
                policy=policy,
                device=cfg.device,
                use_amp=cfg.use_amp,
                fps=cfg.fps,
                single_task=cfg.single_task,
            )

        if keyboard_listener.stop_recording() and recording:
            LOGGER.info(f"Stopping episode {recorded_episodes}")
            dataset.save_episode()
            recorded_episodes += 1
            recording = False

        if keyboard_listener.discard_recording() and recording:
            LOGGER.info("Discarding episode")
            keyboard_listener.events.dict_events["exit_early"] = False
            dataset.clear_episode_buffer()
            recording = False

        if keyboard_listener.exit():
            break

    LOGGER.info("Stop recording")

    if cfg.display_cameras:
        cv2.destroyAllWindows()

    if cfg.push_to_hub:
        dataset.push_to_hub(tags=cfg.tags, private=cfg.private)

    move_robots_to_safe_position(robot)
    keyboard_listener.stop()

    LOGGER.info("Exiting")
    return dataset


@parser.wrap()
def control_robot(cfg: RecordControlConfig):
    """Main function to control the robot and record data.

    Args:
        cfg: The configuration for the recording.
    """
    LOGGER.info(pformat(asdict(cfg)))

    robot = ManipulatorRobot(
        So100RobotConfig(
            leader_arms={
                "main": FeetechMotorsBusConfig(
                    port="/dev/LeRobotLeader",
                    motors={
                        "shoulder_pan": [1, "sts3215"],
                        "shoulder_lift": [2, "sts3215"],
                        "elbow_flex": [3, "sts3215"],
                        "wrist_flex": [4, "sts3215"],
                        "wrist_roll": [5, "sts3215"],
                        "gripper": [6, "sts3215"],
                    },
                ),
            },
            follower_arms={
                "main": FeetechMotorsBusConfig(
                    port="/dev/LeRobotFollower",
                    motors={
                        "shoulder_pan": [1, "sts3215"],
                        "shoulder_lift": [2, "sts3215"],
                        "elbow_flex": [3, "sts3215"],
                        "wrist_flex": [4, "sts3215"],
                        "wrist_roll": [5, "sts3215"],
                        "gripper": [6, "sts3215"],
                    },
                ),
            },
            calibration_dir=f"{SCRIPT_DIR}/.cache/calibration/so100/",
            cameras={
                "wrist": IntelRealSenseCameraConfig(
                    serial_number=145522062152,
                    fps=30,
                    width=640,
                    height=480,
                ),
                "scene": IntelRealSenseCameraConfig(
                    serial_number=251622063326,
                    fps=30,
                    width=640,
                    height=480,
                ),
            },
        ),
    )

    def signal_handler(sig, frame):
        """Handle the SIGINT signal."""
        LOGGER.info("SIGINT received, moving robots to a safe position")
        move_robots_to_safe_position(robot)
        robot.disconnect()
        sys.exit(0)

    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)

    record(robot, cfg)

    if robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()


if __name__ == "__main__":
    control_robot()
