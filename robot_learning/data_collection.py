from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.cameras.configs import IntelRealSenseCameraConfig
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
import pathlib

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)

LOGGER = logging.getLogger(__name__)
max_velocity = np.ones(6) * 360
max_acceleration = np.ones(6) * 1000
initial_positions = np.array([-5, 92, 95, 82, -45, 10])
reset_positions = np.array([-1.5, 190, 180, 46, -45, 10])
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
                serial_number=145522062152, fps=30, width=640, height=480
            ),
            "scene": IntelRealSenseCameraConfig(
                serial_number=251622063326, fps=30, width=640, height=480
            ),
        },
    )
)
robot.connect()


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
    move_arm(robot.leader_arms["main"], reset_positions)
    robot.leader_arms["main"].write("Torque_Enable", 0)
    move_arm(robot.follower_arms["main"], reset_positions)
    robot.follower_arms["main"].write("Torque_Enable", 0)
    robot.disconnect()
    sys.exit(0)


# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

move_arm(robot.leader_arms["main"], initial_positions)
move_arm(robot.follower_arms["main"], initial_positions)


input("Press Enter to start teleop...")

robot.leader_arms["main"].write("Torque_Enable", 0)

rate_limiter = loop_rate_limiters.RateLimiter(30)
while True:
    observation, action = robot.teleop_step(record_data=True)
    # LOGGER.info(observation["observation.state"])
    # LOGGER.info(observation["observation.images.wrist"].shape)
    # LOGGER.info(observation["observation.images.scene"].shape)
    # LOGGER.info(observation["observation.images.wrist"].min().item())
    # LOGGER.info(observation["observation.images.wrist"].max().item())
    cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
    cv2.imshow(
        "RealSense",
        cv2.cvtColor(
            np.hstack(
                [
                    observation["observation.images.wrist"],
                    observation["observation.images.scene"],
                ]
            ),
            cv2.COLOR_BGR2RGB,
        ),
    )
    cv2.waitKey(1)
    rate_limiter.sleep()
