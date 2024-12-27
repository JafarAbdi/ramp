import pathlib

import numpy as np
import pinocchio as pin

from ramp.robot import Robot, RobotState
from ramp.motion_planner import MotionPlanner
from ramp.visualizer import Visualizer

FILE_PATH = pathlib.Path(__file__).parent


def test_robot_state():
    robot = Robot(FILE_PATH / ".." / "robots" / "unitree_h1" / "configs.toml")

    LEFT_ARM_GROUP = "left_arm"
    RIGHT_ARM_GROUP = "right_arm"

    rs = RobotState.from_named_state(robot.robot_model, LEFT_ARM_GROUP, "home")
    assert np.allclose(rs.actuated_qpos, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rs[LEFT_ARM_GROUP] = [0.5, 0.5, 0.5, 0.5]
    assert np.allclose(rs.actuated_qpos, [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    rs[RIGHT_ARM_GROUP] = [-0.5, -0.5, -0.5, -0.5]
    assert np.allclose(rs.actuated_qpos, [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5])
    rs2 = rs.clone()
    assert np.allclose(rs2.actuated_qpos, rs.actuated_qpos)
    rs2[LEFT_ARM_GROUP] = [0.0, 0.0, 0.0, 0.0]
    assert np.allclose(rs2.actuated_qpos, [0.0, 0.0, 0.0, 0.0, -0.5, -0.5, -0.5, -0.5])
    rs3 = rs.clone()
    rs3[LEFT_ARM_GROUP] = [0.0, 0.0, 0.0, 0.2]
    assert np.allclose(rs3.actuated_qpos, [0.0, 0.0, 0.0, 0.2, -0.5, -0.5, -0.5, -0.5])
    assert np.allclose(
        rs3.qpos,
        RobotState.from_actuated_qpos(robot.robot_model, rs3.actuated_qpos).qpos,
    )

    # Contains both continuous and mimic joints
    robot = Robot(FILE_PATH / ".." / "robots" / "kinova" / "configs.toml")
    rs = RobotState.from_named_state(robot.robot_model, "arm", "home")
    assert np.allclose(rs.actuated_qpos, [0.0, 0.0, -3.14, -1.57, 0.0, 0.0, 1.57])
    rs["arm"] = [0.0, 0.1, 0.5, -2.0, 0.0, 0.0, 0.4]
    assert np.allclose(rs.actuated_qpos, [0.0, 0.1, 0.5, -2.0, 0.0, 0.0, 0.4])
    assert np.allclose(
        rs.qpos,
        RobotState.from_actuated_qpos(robot.robot_model, rs.actuated_qpos).qpos,
    )
    rs2 = rs.clone()
    rs2["arm"] = [0.5, 0.0, -3.14, -1.57, 0.0, 0.0, 1.57]
    assert not np.allclose(rs2.actuated_qpos, rs.actuated_qpos)
    # Test continuous joint being updated correctly
    assert np.allclose(rs2.actuated_qpos, rs2["arm"])
