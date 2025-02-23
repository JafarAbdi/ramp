"""Example of using the differential_ik functionality."""

import logging
import pathlib
import time

import zenoh
from rich.logging import RichHandler
from loop_rate_limiters import RateLimiter

from ramp.mujoco_interface import MuJoCoHardwareInterface
from ramp import setup_logging, load_robot_model
from ramp.ik_solver import IKSolver

GROUP_NAME = "arm"

LOGGER = logging.getLogger(__name__)

robot_model = load_robot_model(pathlib.Path("robots/panda_mj/configs.toml"))
mj_interface = MuJoCoHardwareInterface(robot_model, "home")
ik_solver = IKSolver(
    robot_model.model_filename,
    robot_model.base_link,
    robot_model.groups[GROUP_NAME].tcp_link_name,
)
mj_interface.add_mocap("world", ([0.5, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0]))

rate = RateLimiter(25, warn=False)
while True:
    rs = mj_interface.read()
    mocap = mj_interface.mocap()
    target_joint_positions = ik_solver.solve(
        list(mocap.pos) + [mocap.quat[1], mocap.quat[2], mocap.quat[3], mocap.quat[0]],
        rs.actuated_qpos(),
    )
    if target_joint_positions is None:
        LOGGER.info("IK failed")
    else:
        mj_interface.ctrl(
            mj_interface.as_mj_ctrl(robot_model.joint_names, target_joint_positions)
        )
    rate.sleep()
