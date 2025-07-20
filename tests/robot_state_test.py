import pathlib

import numpy as np

from ramp import load_robot_model, RobotState

FILE_PATH = pathlib.Path(__file__).parent


def test_robot_state():
    robot_model = load_robot_model(
        FILE_PATH / ".." / "robots" / "unitree_h1" / "configs.toml"
    )

    LEFT_ARM_GROUP = "left_arm"
    RIGHT_ARM_GROUP = "right_arm"

    rs = RobotState.from_named_state(robot_model, LEFT_ARM_GROUP, "home")
    assert np.allclose(rs.actuated_qpos(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rs.set_group_qpos(LEFT_ARM_GROUP, [0.5, 0.5, 0.5, 0.5])
    assert np.allclose(rs.actuated_qpos(), [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    rs.set_group_qpos(RIGHT_ARM_GROUP, [-0.5, -0.5, -0.5, -0.5])
    assert np.allclose(rs.actuated_qpos(), [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5])
    rs2 = rs.clone()
    assert np.allclose(rs2.actuated_qpos(), rs.actuated_qpos())
    rs2.set_group_qpos(LEFT_ARM_GROUP, [0.0, 0.0, 0.0, 0.0])
    assert np.allclose(
        rs2.actuated_qpos(), [0.0, 0.0, 0.0, 0.0, -0.5, -0.5, -0.5, -0.5]
    )
    rs3 = rs.clone()
    rs3.set_group_qpos(LEFT_ARM_GROUP, [0.0, 0.0, 0.0, 0.2])
    assert np.allclose(
        rs3.actuated_qpos(), [0.0, 0.0, 0.0, 0.2, -0.5, -0.5, -0.5, -0.5]
    )
    assert np.allclose(
        rs3.qpos,
        RobotState.from_actuated_qpos(robot_model, rs3.actuated_qpos()).qpos,
    )

    # Contains both continuous and mimic joints
    robot_model = load_robot_model(FILE_PATH / "double_pendulum.toml")
    rs = RobotState(robot_model)
    assert np.allclose(rs.actuated_qpos(), [0.0, 0.0])
    rs.set_group_qpos("arm", [0.2, 0.1])
    assert np.allclose(rs.actuated_qpos(), [0.2, 0.1])
    assert np.allclose(
        rs.qpos,
        RobotState.from_actuated_qpos(robot_model, rs.actuated_qpos()).qpos,
    )
    rs2 = rs.clone()
    rs2.set_group_qpos("arm", [0.5, 0.1])
    assert not np.allclose(rs2.actuated_qpos(), rs.actuated_qpos())
    # Test continuous joint being updated correctly
    assert np.allclose(rs2.actuated_qpos(), rs2.group_qpos("arm"))
    assert np.allclose(
        rs2.qpos[robot_model.mimic_joint_indices], [-0.05]
    )  # Mimic joint updated

