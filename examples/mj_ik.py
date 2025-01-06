"""Example of using the differential_ik functionality."""

import logging
import pathlib
import time

from rich.logging import RichHandler

# from mujoco_simulator_py.mujoco_interface import MuJoCoInterface
from ramp import load_robot_model, setup_logging
from ramp.ik_solver import IKSolver
from ramp.hardware_interface import MuJoCoHardwareInterface

setup_logging()
LOGGER = logging.getLogger(__name__)

GROUP_NAME = "arm"
# robot_model = load_robot_model(pathlib.Path("robots/rrr_mj/configs.toml"))
robot_model = load_robot_model(pathlib.Path("robots/panda_mj/configs.toml"))
mj_interface = MuJoCoHardwareInterface(robot_model, "home")

ik_solver = IKSolver(
    robot_model.model_filename,
    robot_model.base_link,
    robot_model.groups[GROUP_NAME].tcp_link_name,
)

for target_pose in [
    [-0.4, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0],
    [0.4, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0],
]:
    robot_state = mj_interface.read()
    target_joint_positions = ik_solver.solve(
        target_pose,
        robot_state.actuated_qpos,
    )
    if target_joint_positions is None:
        LOGGER.info("IK failed")
    else:
        LOGGER.info(f"IK succeeded: {target_joint_positions}")
        mj_interface.write(robot_model[GROUP_NAME].joints, target_joint_positions)
    time.sleep(1.0)
