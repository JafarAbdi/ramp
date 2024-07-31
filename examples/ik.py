"""Example of using the differential_ik functionality."""

import sys
import logging
import numpy as np
import pathlib

from rich.logging import RichHandler

from ramp.robot import Robot
from ramp.visualizer import Visualizer
from ramp.ik_solver import IKSolver
from ramp.constants import GROUP_NAME


import logging
import pathlib
import time

import zenoh
from rich.logging import RichHandler

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


class IKDemo:
    def __init__(self, config_name: str):
        self.robot = Robot(pathlib.Path(f"robots/{config_name}/configs.toml"))
        self.visualizer = Visualizer(self.robot)
        self.reset()

    def visualize(self, joint_positions):
        """Visualize a joint position using meshcat."""
        self.visualizer.robot_state(joint_positions)

    def reset(self):
        self.visualizer.robot_state(self.robot.named_states["home"])

    # TODO: Refactor to make this nicer, I hate this so much
    def seed(self):
        if self.robot.groups[GROUP_NAME].gripper is None:
            return self.robot.named_states["home"]
        return self.robot.named_states["home"][:-1]

    def run(self):
        target_pose = [0.4, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0]
        input("Press Enter to start IK using differential_ik solver")
        target_joint_positions = self.robot.differential_ik(
            target_pose,
            self.robot.named_states["home"],
            iteration_callback=self.visualize,
        )
        if target_joint_positions is None:
            LOGGER.info("IK failed")
        else:
            LOGGER.info(f"IK succeeded: {target_joint_positions}")
            LOGGER.info(
                f"TCP Pose for target joint positions: {self.robot.get_frame_pose(target_joint_positions, self.robot.groups['arm'].tcp_link_name)}",
            )

        self.reset()
        input("Press Enter to start IK using trac-ik solver")
        ik_solver = IKSolver(
            self.robot.model_filename,
            self.robot.base_link,
            self.robot.groups[GROUP_NAME].tcp_link_name,
        )
        target_joint_positions = ik_solver.solve(
            target_pose,
            self.seed(),
        )
        if target_joint_positions is None:
            LOGGER.info("IK failed")
        else:
            target_joint_positions = np.concatenate(
                (
                    [
                        target_joint_positions,
                        (
                            [self.robot.named_states["home"][-1]]
                            if self.robot.groups[GROUP_NAME].gripper
                            else []
                        ),
                    ]
                )
            )
            LOGGER.info(f"IK succeeded: {target_joint_positions}")
            LOGGER.info(
                f"TCP Pose for target joint positions: {self.robot.get_frame_pose(target_joint_positions, self.robot.groups['arm'].tcp_link_name)}",
            )
            self.visualize(target_joint_positions)


def main():
    configs = ["panda", "rrr", "kinova", "ur5e", "fr3_robotiq"]
    for config in configs:
        LOGGER.info(f"Running IK for {config}")
        input(f"Press Enter to start {config} IK")
        demo = IKDemo(config)
        demo.run()


if __name__ == "__main__":
    main()
