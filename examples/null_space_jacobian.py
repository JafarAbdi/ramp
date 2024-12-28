import logging
import pathlib

import sys
import casadi
import numpy as np
import pinocchio as pin
from rich.logging import RichHandler

from ramp import load_robot_model, CasADiRobot, RobotState, Visualizer, setup_logging

setup_logging()
LOGGER = logging.getLogger(__name__)


# Define variables:
# θ1, θ2, θ3: joint angles
# l1, l2, l3: link lengths

# End-effector position:
# x = 0 (always, since rotation is around x-axis)
# y = l1*sin(θ1) + l2*sin(θ1+θ2) + l3*sin(θ1+θ2+θ3)
# z = l1*cos(θ1) + l2*cos(θ1+θ2) + l3*cos(θ1+θ2+θ3)

# Jacobian J is a 3x3 matrix:
# J = [∂x/∂θ1  ∂x/∂θ2  ∂x/∂θ3]
#     [∂y/∂θ1  ∂y/∂θ2  ∂y/∂θ3]
#     [∂z/∂θ1  ∂z/∂θ2  ∂z/∂θ3]

# Calculating each element:

# ∂x/∂θ1 = ∂x/∂θ2 = ∂x/∂θ3 = 0 (x is always 0)

# ∂y/∂θ1 = l1*cos(θ1) + l2*cos(θ1+θ2) + l3*cos(θ1+θ2+θ3)
# ∂y/∂θ2 = l2*cos(θ1+θ2) + l3*cos(θ1+θ2+θ3)
# ∂y/∂θ3 = l3*cos(θ1+θ2+θ3)

# ∂z/∂θ1 = -l1*sin(θ1) - l2*sin(θ1+θ2) - l3*sin(θ1+θ2+θ3)
# ∂z/∂θ2 = -l2*sin(θ1+θ2) - l3*sin(θ1+θ2+θ3)
# ∂z/∂θ3 = -l3*sin(θ1+θ2+θ3)


# This is equivalent to pinocchio.computeFrameJacobian with reference_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
def calculate_jacobian(theta):
    t1, t2, t3 = theta[:3]
    l1 = l2 = l3 = 1.0

    J = np.array(
        [
            [0, 0, 0],  # x-component (always 0 in ZY plane)
            [
                l1 * np.cos(t1) + l2 * np.cos(t1 + t2) + l3 * np.cos(t1 + t2 + t3),
                l2 * np.cos(t1 + t2) + l3 * np.cos(t1 + t2 + t3),
                l3 * np.cos(t1 + t2 + t3),
            ],
            [
                -l1 * np.sin(t1) - l2 * np.sin(t1 + t2) - l3 * np.sin(t1 + t2 + t3),
                -l2 * np.sin(t1 + t2) - l3 * np.sin(t1 + t2 + t3),
                -l3 * np.sin(t1 + t2 + t3),
            ],
        ],
    )

    return J


robot_model = load_robot_model(pathlib.Path("robots/planar_rrr/configs.toml"))

theta = np.array([np.pi / 4, np.pi / 3, np.pi / 6])
robot_state = RobotState.from_actuated_qpos(robot_model, [0.0, 0.0, 0.0])
LOGGER.info(
    robot_state.jacobian(
        "end_effector",
        reference_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )[:3, :3],
)
robot_state = RobotState.from_actuated_qpos(robot_model, theta)
LOGGER.info(
    robot_state.jacobian(
        "end_effector",
        reference_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )[:3, :3],
)
LOGGER.info(calculate_jacobian(np.array([0.0, 0.0, 0.0])))
LOGGER.info(calculate_jacobian(theta))

LOGGER.info("Calculating Jacobian Pseudo-Inverse")
robot_state = RobotState.from_actuated_qpos(robot_model, theta)
J = robot_state.jacobian(
    "end_effector",
    reference_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
)[:3, :3]
J_pinv = np.linalg.pinv(J)

LOGGER.info(J)
LOGGER.info(J_pinv)
LOGGER.info(f"Null-Space Jacobian: {np.eye(3) - J_pinv @ J}")

# Primary task: Move end-effector in y direction
x_dot = np.array([0.0, 0.1, 0.0])

LOGGER.info(f"J+ @ x_dot: {J_pinv @ x_dot}")

# Secondary task: Try to move the second joint towards pi/2
theta_desired = np.array([0, np.pi / 2, 0])
k = 0.1  # Gain for the secondary task
z = k * (theta_desired - theta[:3])

# Calculate joint velocities
theta_dot = J_pinv @ x_dot + (np.eye(3) - J_pinv @ J) @ z
LOGGER.info(f"Joint velocities: {theta_dot}")
LOGGER.info(
    f"J * (I - J_pinv @ J) @ z = eef_velocity should be zero: {J @ (np.eye(3) - J_pinv @ J) @ z}",
)

group_name = "arm"
tcp_name = robot_model[group_name].tcp_link_name
if len(sys.argv) > 1 and sys.argv[1] == "visualize":
    visualizer = Visualizer(robot_model)
    robot_state = RobotState.from_actuated_qpos(robot_model, [0.0, 0.0, 0.0])
    visualizer.robot_state(robot_state)
    visualizer.frame(
        tcp_name,
        robot_state.get_frame_pose(tcp_name).np,
    )

casadi_robot = CasADiRobot(robot_model)
LOGGER.info(casadi_robot.data.oMf[robot_model.model.getFrameId(tcp_name)])
LOGGER.info(casadi.simplify(casadi_robot.jacobian(tcp_name, pin.LOCAL_WORLD_ALIGNED)))
