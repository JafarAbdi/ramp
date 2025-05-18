from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.cameras.configs import IntelRealSenseCameraConfig
from functools import partial
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
import cv2
import time
import numpy as np
import loop_rate_limiters
from ramp.trajectory_smoothing import generate_time_optimal_trajectory_from_waypoints
import sys
import signal
import logging
import rich
from rich.logging import RichHandler
import os
from lerobot.common.robot_devices.motors.feetech import (
    LOWER_BOUND_LINEAR,
    UPPER_BOUND_LINEAR,
    GRIPPER_JOINT_NORMALIZER,
    GRIPPER_JOINT_DENORMALIZER,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)

LOGGER = logging.getLogger(__name__)
max_velocity = np.ones(6) * 360
max_acceleration = np.ones(6) * 1000
gripper_command = 100
initial_positions = np.array([-5, 92, 95, 82, -50, gripper_command])
reset_positions = np.array([0.0, 190.0, 180.0, 30.0, -50.0, 25.0])

from threading import Lock, Thread

from sshkeyboard import listen_keyboard


# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"


class KeyboardListener:
    """A class to listen to keyboard events."""

    def __init__(self):
        """Initialize the keyboard listener."""
        self._mutex = Lock()
        self._toggle_gripper = False
        self._toggle_torque = False
        self._start = False
        self._keyboard_listener_thread = Thread(
            target=listen_keyboard,
            args=(self._on_key_press,),
            daemon=True,
        )
        self._keyboard_listener_thread.start()

        LOGGER.info(f"{CYAN}Keyboard Controls:{RESET}")
        LOGGER.info(f"  {GREEN}space{RESET}: Toggle gripper.")
        LOGGER.info(f"  {GREEN}e{RESET}: Toggle torque.")
        LOGGER.info(f"  {GREEN}s{RESET}: Start.")

    def _on_key_press(self, key):
        """Handle key press from the keyboard."""
        LOGGER.info(f"Key pressed: {key}")
        with self._mutex:
            if key == "space":
                self._toggle_gripper = True
            elif key == "e":
                self._toggle_torque = True
            elif key == "s":
                self._start = True

    @property
    def toggle_gripper(self) -> bool:
        """Get the success request."""
        toggle, self._toggle_gripper = (self._toggle_gripper, False)
        return toggle

    @property
    def toggle_torque(self) -> bool:
        """Get the success request."""
        toggle, self._toggle_torque = (self._toggle_torque, False)
        return toggle

    @property
    def start(self) -> bool:
        """Get the success request."""
        start, self._start = (self._start, False)
        return start


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
                mock=True,
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
        calibration_dir=".cache/calibration/so100/",
        cameras={},
    )
)
robot.connect()
listener = KeyboardListener()


def move_arm(arm, positions):
    trajectory = generate_time_optimal_trajectory_from_waypoints(
        [arm.read("Present_Position"), positions],
        max_velocity,
        max_acceleration,
    )

    LOGGER.info(f"Generated trajectory with {len(trajectory)} points")
    t = 0.0
    for trajectory_point, time_from_start in trajectory:
        arm.write("Goal_Position", trajectory_point)
        time.sleep(time_from_start - t)
        t = time_from_start


def signal_handler(sig, frame):
    move_arm(robot.follower_arms["main"], reset_positions)
    robot.follower_arms["main"].write("Torque_Enable", 0)
    robot.disconnect()
    sys.exit(0)


# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

move_arm(robot.follower_arms["main"], initial_positions)

# while not listener.start:
#     LOGGER.info("Waiting for start")
#     time.sleep(1.0)


# robot.follower_arms["main"].write("Torque_Enable", 0)

rate_limiter = loop_rate_limiters.RateLimiter(30)
while True:
    observation, action = robot.teleop_step(record_data=True)
    LOGGER.info(observation["observation.state"])
    if listener.toggle_gripper:
        gripper_command = 100 - gripper_command
        robot.follower_arms["main"].write(
            "Goal_Position",
            np.concatenate((observation["observation.state"][:-1], [gripper_command])),
        )
    if listener.toggle_torque:
        robot.follower_arms["main"].write(
            "Torque_Enable", 1 - robot.follower_arms["main"].read("Torque_Enable")
        )

    # move_arm(robot.follower_arms["main"], initial_positions)
    # LOGGER.info(observation["observation.images.wrist"].shape)
    # LOGGER.info(observation["observation.images.scene"].shape)
    # LOGGER.info(observation["observation.images.wrist"].min().item())
    # LOGGER.info(observation["observation.images.wrist"].max().item())
    rate_limiter.sleep()
