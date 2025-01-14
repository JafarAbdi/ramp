"""Example of using the differential_ik functionality."""

import sys
import logging
import numpy as np
import pathlib

from rich.logging import RichHandler

from ramp import load_robot_model, RobotState, Visualizer, setup_logging
from ramp.ik_solver import IKSolver


import logging
import pathlib
import time

import zenoh
from rich.logging import RichHandler


setup_logging()
LOGGER = logging.getLogger(__name__)

GROUP_NAME = "arm"


class FakeVisualizer:
    def robot_state(self, robot_state):
        pass


class IKDemo:
    def __init__(self, config_name: str, visualize: bool):
        self.robot_model = load_robot_model(
            pathlib.Path(f"robots/{config_name}/configs.toml")
        )
        self.prompt = visualize
        self.visualizer = (
            Visualizer(self.robot_model) if visualize else FakeVisualizer()
        )
        self.initial_state = RobotState.from_named_state(
            self.robot_model,
            GROUP_NAME,
            "home",
        )
        self.reset()

    def visualize(self, robot_state):
        """Visualize a joint position using meshcat."""
        self.visualizer.robot_state(robot_state)

    def reset(self):
        self.visualize(self.initial_state)

    def run(self):
        target_pose = [0.4, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0]
        if self.prompt:
            input("Press Enter to start IK using differential_ik solver")
        robot_state = self.initial_state.clone()
        if not robot_state.differential_ik(
            GROUP_NAME,
            target_pose,
            iteration_callback=self.visualize,
        ):
            LOGGER.info("IK failed")
        else:
            LOGGER.info(f"IK succeeded: {robot_state}")
            LOGGER.info(
                f"TCP Pose for target joint positions: {robot_state.get_frame_pose(self.robot_model[GROUP_NAME].tcp_link_name)}",
            )

        if self.prompt:
            input("Press Enter to start IK using trac-ik solver")
        self.reset()
        ik_solver = IKSolver(
            self.robot_model.model_filename,
            self.robot_model.base_link,
            self.robot_model[GROUP_NAME].tcp_link_name,
        )
        target_joint_positions = ik_solver.solve(
            target_pose, self.initial_state.actuated_qpos()
        )
        assert target_joint_positions is not None, "IK failed"
        robot_state = self.initial_state.clone()
        robot_state.set_group_qpos(GROUP_NAME, target_joint_positions)
        LOGGER.info(f"IK succeeded: {target_joint_positions}")
        LOGGER.info(
            f"TCP Pose for target joint positions: {robot_state.get_frame_pose(self.robot_model[GROUP_NAME].tcp_link_name)}",
        )
        self.visualize(robot_state)


configs = ["panda", "rrr", "kinova", "ur5e", "fr3_robotiq"]
for config in configs:
    LOGGER.info(f"Running IK for {config}")
    demo = IKDemo(config, len(sys.argv) > 1 and sys.argv[1] == "visualize")
    demo.run()
