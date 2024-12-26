import pathlib

import numpy as np
import pinocchio as pin

from ramp.robot import Robot, RobotState, GroupState
from ramp.motion_planner import MotionPlanner
from ramp.visualizer import Visualizer

FILE_PATH = pathlib.Path(__file__).parent
robot = Robot(FILE_PATH / ".." / "robots" / "unitree_h1" / "configs.toml")


def test_robot_state():
    LEFT_ARM_GROUP = "left_arm"
    RIGHT_ARM_GROUP = "right_arm"

    rs = RobotState(
        robot.robot_model, robot.robot_model.named_state(LEFT_ARM_GROUP, "home")
    )
    assert np.allclose(rs.actuated_qpos, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rs[LEFT_ARM_GROUP] = [0.5, 0.5, 0.5, 0.5]
    assert np.allclose(rs.actuated_qpos, [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    rs[RIGHT_ARM_GROUP] = [-0.5, -0.5, -0.5, -0.5]
    assert np.allclose(rs.actuated_qpos, [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5])
    rs2 = rs.clone()
    assert np.allclose(rs2.actuated_qpos, rs.actuated_qpos)
    rs2[LEFT_ARM_GROUP] = [0.0, 0.0, 0.0, 0.0]
    assert np.allclose(rs2.actuated_qpos, [0.0, 0.0, 0.0, 0.0, -0.5, -0.5, -0.5, -0.5])
    rs3 = rs.clone(GroupState(LEFT_ARM_GROUP, [0.0, 0.0, 0.0, 0.2]))
    assert np.allclose(rs3.actuated_qpos, [0.0, 0.0, 0.0, 0.2, -0.5, -0.5, -0.5, -0.5])
