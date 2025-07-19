"""A wrapper around time_optimal_trajectory_generation."""

import logging

import numpy as np
import time_optimal_trajectory_generation_py as totg

from ramp.robot_model import RobotModel
from ramp.robot_state import RobotState

LOGGER = logging.getLogger(__name__)


def generate_time_optimal_trajectory_from_waypoints(
    waypoints: list[list[float]],
    max_velocity: list[float],
    max_acceleration: list[float],
    resample_dt=0.1,
) -> list[tuple[list[float], float]] | None:
    """Parameterize the trajectory using Time Optimal Trajectory Generation http://www.golems.org/node/1570.

    Args:
        waypoints: The waypoints to parameterize.
        max_velocity: The maximum velocity for each joint.
        max_acceleration: The maximum acceleration for each joint.
        resample_dt: The resampling time step.

    Returns:
        The parameterized trajectory as a list of (robot_state, time_from_start).
    """
    # The intermediate waypoints of the input path need to be blended so that the entire path is differentiable.
    # This constant defines the maximum deviation allowed at those intermediate waypoints, in radians for revolute joints,
    # or meters for prismatic joints.
    max_deviation = 0.1
    trajectory = totg.Trajectory(
        totg.Path(
            waypoints,
            max_deviation,
        ),
        max_velocity,
        max_acceleration,
    )
    if not trajectory.isValid():
        LOGGER.error("Failed to parameterize trajectory")
        return None
    duration = trajectory.getDuration()
    parameterized_trajectory = []
    for t in np.append(np.arange(0.0, duration, resample_dt), duration):
        parameterized_trajectory.append((trajectory.getPosition(t), t))
    return parameterized_trajectory


def generate_time_optimal_trajectory(
    robot_model: RobotModel,
    group_name: str,
    waypoints: list[RobotState],
    resample_dt=0.1,
) -> list[tuple[RobotState, float]] | None:
    """Parameterize the trajectory using Time Optimal Trajectory Generation http://www.golems.org/node/1570.

    Args:
        robot_model: The robot model.
        group_name: The group name.
        waypoints: The waypoints to parameterize.
        resample_dt: The resampling time step.

    Returns:
        The parameterized trajectory as a list of (robot_state, time_from_start).
    """
    # The intermediate waypoints of the input path need to be blended so that the entire path is differentiable.
    # This constant defines the maximum deviation allowed at those intermediate waypoints, in radians for revolute joints,
    # or meters for prismatic joints.
    if robot_model.acceleration_limits.size == 0:
        raise ValueError(
            "Acceleration limits are required for trajectory parameterization"
            "Make sure to specify acceleration limits in the robot model"
            "\n[acceleration_limits]\n"
            + "\n".join(
                [f"{joint_name} = X" for joint_name in robot_model.joint_names],
            ),
        )
    parameterized_trajectory = generate_time_optimal_trajectory_from_waypoints(
        [waypoint.group_qpos(group_name) for waypoint in waypoints],
        robot_model.velocity_limits,
        robot_model.acceleration_limits,
        resample_dt,
    )

    trajectory = []
    for trajectory_point, t in parameterized_trajectory:
        rs = waypoints[0].clone()
        rs.set_group_qpos(group_name, trajectory_point)
        trajectory.append((rs, t))
    return trajectory
