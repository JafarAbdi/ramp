"""Example of using the differential_ik functionality."""

import logging
import pathlib
import time

import zenoh
from rich.logging import RichHandler

from mujoco_simulator_py.mujoco_interface import MuJoCoInterface
from ramp.robot import Robot
from ramp.mj_robot import MjRobot
from ramp.ik_solver import IKSolver

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
LOGGER = logging.getLogger(__name__)


zenoh.init_log_from_env_or("error")

GROUP_NAME = "arm"
# > robot = Robot(pathlib.Path("robots/rrr_mj/configs.toml"))
robot = Robot(pathlib.Path("robots/panda_mj/configs.toml"))

# Should this wrap MujocoInterface?
mj_robot = MjRobot(robot.model_filename, robot.groups)

mujoco_interface = MuJoCoInterface()
mujoco_interface.reset(model_filename=mj_robot.model_filename)

ik_solver = IKSolver(
    robot.model_filename,
    robot.base_link,
    robot.groups[GROUP_NAME].tcp_link_name,
)

for target_pose in [
    [-0.4, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0],
    [0.4, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0],
]:
    qpos = mj_robot.from_mj_joint_positions(mujoco_interface.qpos())
    target_joint_positions = ik_solver.solve(
        target_pose,
        qpos,
    )
    if target_joint_positions is None:
        LOGGER.info("IK failed")
    else:
        LOGGER.info(f"IK succeeded: {target_joint_positions}")
        mujoco_interface.ctrl(
            mj_robot.as_mj_ctrl(robot.groups[GROUP_NAME].joints, target_joint_positions)
        )
    time.sleep(1.0)
