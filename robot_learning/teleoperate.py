"""This script teleoperates the SO100 robot using a keyboard listener."""

import logging
import os
import pathlib
import signal
import sys

import cv2
import loop_rate_limiters
import numpy as np
from lerobot.common.robot_devices.cameras.configs import IntelRealSenseCameraConfig
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
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
robot.connect()


def signal_handler(sig, frame):
    """Handle the SIGINT signal."""
    move_robots_to_safe_position(robot)
    robot.disconnect()
    sys.exit(0)


# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

move_robots_to_initial_position(robot)

input("Press Enter to start teleop...")
keyboard_listener = KeyboardListener()

robot.leader_arms["main"].write("Torque_Enable", 0)

rate_limiter = loop_rate_limiters.RateLimiter(30)
while True:
    observation, action = robot.teleop_step(
        record_data=True,
        events=keyboard_listener.events.dict_events,
    )
    LOGGER.info(observation["observation.state"])
    # > LOGGER.info(observation["observation.images.wrist"].shape)
    # > LOGGER.info(observation["observation.images.scene"].shape)
    # > LOGGER.info(observation["observation.images.wrist"].min().item())
    # > LOGGER.info(observation["observation.images.wrist"].max().item())
    cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
    cv2.imshow(
        "RealSense",
        cv2.cvtColor(
            np.hstack(
                [
                    observation["observation.images.wrist"],
                    observation["observation.images.scene"],
                ],
            ),
            cv2.COLOR_BGR2RGB,
        ),
    )
    cv2.waitKey(1)
    rate_limiter.sleep()
