"""Inverse kinematics using CasADi for a given robot configuration."""

import sys
import logging
import pathlib

import casadi
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
from rich.logging import RichHandler

from ramp import load_robot_model, CasADiRobot, RobotState, Visualizer, setup_logging

setup_logging()
LOGGER = logging.getLogger(__name__)

GROUP_NAME = "arm"


class FakeVisualizer:
    def robot_state(self, robot_state):
        pass

    def frame(self, name, pose):
        pass


def ik(config_name: str, visualize: bool):
    """Inverse kinematics using CasADi for a given robot configuration."""
    robot_model = load_robot_model(pathlib.Path(f"robots/{config_name}/configs.toml"))
    casadi_robot = CasADiRobot(robot_model)
    visualizer = Visualizer(robot_model) if visualize else FakeVisualizer()

    transform_target_to_world = pin.SE3(
        pin.utils.rpyToMatrix(np.pi, 0, np.pi / 2),
        np.array([-0.5, 0.1, 0.2]),
    )

    visualizer.frame("target", transform_target_to_world.np)

    def callback(q: np.ndarray):
        robot_state = RobotState(robot_model, q)
        visualizer.frame(
            "current",
            robot_state.get_frame_pose(robot_model[GROUP_NAME].tcp_link_name).np,
        )
        visualizer.robot_state(robot_state)

    error_tool = casadi.Function(
        "etool",
        [casadi_robot.q],
        [
            cpin.log6(
                casadi_robot.data.oMf[
                    robot_model.model.getFrameId(robot_model[GROUP_NAME].tcp_link_name)
                ].inverse()
                * cpin.SE3(transform_target_to_world),
            ).vector,
        ],
    )

    opti = casadi.Opti()
    var_q = opti.variable(robot_model.model.nq)

    # Add a regularization term to keep the solution close to the neutral pose of the robot
    # This helps with convergence and numerical stability
    regularization = 1e-6 * casadi.sumsqr(var_q - pin.neutral(robot_model.model))
    totalcost = casadi.sumsqr(error_tool(var_q)) + regularization

    constraints = casadi.vertcat(
        *[
            var_q[idx, 0] ** 2 + var_q[idx + 1, 0] ** 2 - 1
            for idx in robot_model.continuous_joint_indices
        ],
    )
    if constraints.shape[0] > 0:
        opti.subject_to(constraints == 0)
    opti.set_initial(var_q, pin.neutral(robot_model.model))
    opti.minimize(totalcost)
    opti.solver("ipopt")  # select the backend solver
    opti.callback(lambda i: callback(opti.debug.value(var_q)))

    # Caution: in case the solver does not converge, we are picking the candidate values
    # at the last iteration in opti.debug, and they are NO guarantee of what they mean.
    try:
        opti.solve_limited()
        sol_q = opti.value(var_q)
        robot_state = RobotState(robot_model, sol_q)
        visualizer.robot_state(robot_state)
    except Exception:
        sol_q = opti.debug.value(var_q)
        LOGGER.exception(f"ERROR in convergence, plotting debug info {sol_q=}.")


for config_name in ["panda", "kinova", "ur5e", "fr3_robotiq"]:
    visualize = "visualize" in sys.argv
    if visualize:
        input(f"Press Enter to start {config_name} IK")
    ik(config_name, visualize)
