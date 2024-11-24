"""Example of using the differential_ik functionality."""

import logging
import pathlib
import time

import zenoh
from rich.logging import RichHandler
from loop_rate_limiters import RateLimiter

from mujoco_simulator_py.mujoco_interface import MuJoCoInterface
from ramp.robot import Robot
from ramp.mj_robot import MjRobot
from ramp.ik_solver import IKSolver
from ramp.constants import GROUP_NAME

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
LOGGER = logging.getLogger(__name__)

zenoh.init_log_from_env_or("error")

robot = Robot(pathlib.Path("robots/panda_mj/configs.toml"))
# robot = Robot(pathlib.Path("robots/ur5e/configs.toml"))
# robot = Robot(pathlib.Path("robots/kinova/configs.toml"))
mj_robot = MjRobot(robot.model_filename, robot.groups)
ik_solver = IKSolver(
    robot.model_filename,
    robot.base_link,
    robot.groups[GROUP_NAME].tcp_link_name,
)

mujoco_interface = MuJoCoInterface()
mujoco_interface.reset(keyframe="home")

rate = RateLimiter(25, warn=False)
while True:
    mocap = mujoco_interface.mocap()
    target_joint_positions = ik_solver.ik(
        mocap[0] + [mocap[1][1], mocap[1][0], mocap[1][2], mocap[1][3]],
        mj_robot.from_mj_joint_positions(mujoco_interface.qpos()),
    )
    if target_joint_positions is None:
        LOGGER.info("IK failed")
    else:
        mujoco_interface.ctrl(
            mj_robot.as_mj_ctrl(mj_robot.actuator_names.keys(), target_joint_positions)
        )
    rate.sleep()
