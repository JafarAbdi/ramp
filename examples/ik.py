"""Example of using the differential_ik functionality."""

import sys
import logging
import numpy as np
import pathlib

from rich.logging import RichHandler

from ramp.robot import Robot, RobotState
from ramp.visualizer import Visualizer
from ramp.ik_solver import IKSolver


import logging
import pathlib
import time

import zenoh
from rich.logging import RichHandler

from mujoco_simulator_py.mujoco_interface import MuJoCoInterface

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
LOGGER = logging.getLogger(__name__)

GROUP_NAME = "arm"


class IKDemo:
    def __init__(self, config_name: str):
        self.robot = Robot(pathlib.Path(f"robots/{config_name}/configs.toml"))
        self.visualizer = Visualizer(self.robot)
        self.initial_state = RobotState.from_named_state(
            self.robot.robot_model,
            GROUP_NAME,
            "home",
        )
        self.reset()

    def visualize(self, robot_state):
        """Visualize a joint position using meshcat."""
        self.visualizer.robot_state(robot_state)

    def reset(self):
        self.visualizer.robot_state(self.initial_state)

    def run(self):
        target_pose = [0.4, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0]
        input("Press Enter to start IK using differential_ik solver")
        target_joint_positions = self.robot.differential_ik(
            GROUP_NAME,
            target_pose,
            self.initial_state,
            iteration_callback=self.visualize,
        )
        if target_joint_positions is None:
            LOGGER.info("IK failed")
        else:
            LOGGER.info(f"IK succeeded: {target_joint_positions}")
            LOGGER.info(
                f"TCP Pose for target joint positions: {self.robot.get_frame_pose(target_joint_positions, self.robot.robot_model[GROUP_NAME].tcp_link_name)}",
            )

        input("Press Enter to start IK using trac-ik solver")
        self.reset()
        ik_solver = IKSolver(
            self.robot.model_filename,
            self.robot.base_link,
            self.robot.robot_model[GROUP_NAME].tcp_link_name,
        )
        target_joint_positions = ik_solver.solve(
            target_pose, self.initial_state.actuated_qpos
        )
        if target_joint_positions is None:
            LOGGER.info("IK failed")
        else:
            robot_state = self.initial_state.clone()
            robot_state[GROUP_NAME] = target_joint_positions
            LOGGER.info(f"IK succeeded: {target_joint_positions}")
            LOGGER.info(
                f"TCP Pose for target joint positions: {self.robot.get_frame_pose(robot_state, self.robot.robot_model[GROUP_NAME].tcp_link_name)}",
            )
            self.visualize(robot_state)


def main():
    configs = ["panda", "rrr", "kinova", "ur5e", "fr3_robotiq"]
    for config in configs:
        LOGGER.info(f"Running IK for {config}")
        input(f"Press Enter to start {config} IK")
        demo = IKDemo(config)
        demo.run()


if __name__ == "__main__":
    main()
