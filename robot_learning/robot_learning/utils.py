"""Utility functions for controlling the robot arms."""

import logging
import time

from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

from ramp.trajectory_smoothing import generate_time_optimal_trajectory_from_waypoints

from .constants import (
    INITIAL_POSITIONS,
    MAX_ACCELERATION,
    MAX_VELOCITY,
    RESET_POSITIONS,
)

LOGGER = logging.getLogger(__name__)


def move_arm(arm, positions):
    """Move a robot arm to the specified positions."""
    trajectory = generate_time_optimal_trajectory_from_waypoints(
        [arm.read("Present_Position"), positions],
        MAX_VELOCITY,
        MAX_ACCELERATION,
    )

    LOGGER.info(f"Generated trajectory with {len(trajectory)} points")
    t = 0.0
    for trajectory_point, time_from_start in trajectory:
        arm.write("Goal_Position", trajectory_point)
        time.sleep(time_from_start - t)
        t = time_from_start


def move_robots_to_safe_position(robot: ManipulatorRobot):
    """Move the robot arms to a safe position."""
    move_arm(robot.leader_arms["main"], RESET_POSITIONS)
    robot.leader_arms["main"].write("Torque_Enable", 0)
    move_arm(robot.follower_arms["main"], RESET_POSITIONS)
    robot.follower_arms["main"].write("Torque_Enable", 0)


def move_robots_to_initial_position(robot: ManipulatorRobot):
    """Move the robot arms to the initial position."""
    move_arm(robot.leader_arms["main"], INITIAL_POSITIONS)
    move_arm(robot.follower_arms["main"], INITIAL_POSITIONS)
