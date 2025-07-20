import pathlib
from ramp.compute_disable_collisions import (
    adjacent_collisions,
    DisabledReason,
)
from ramp import load_robot_model


FILE_PATH = pathlib.Path(__file__).parent


def test_disable_collisions():
    robot_model = load_robot_model(
        FILE_PATH / ".." / "robots" / "rrr" / "rrr.urdf.xacro"
    )
    assert set(adjacent_collisions(robot_model)) == set(
        [
            ("base_link", "link1", DisabledReason.ADJACENT),
            ("link1", "link2", DisabledReason.ADJACENT),
            ("link2", "link3", DisabledReason.ADJACENT),
            ("link3", "end_effector", DisabledReason.ADJACENT),
        ]
    )
    robot_model = load_robot_model(
        FILE_PATH / ".." / "robots" / "acrobot" / "acrobot.xml"
    )
    assert set(adjacent_collisions(robot_model)) == set(
        [("upper_link", "lower_link", DisabledReason.ADJACENT)]
    )

    robot_model = load_robot_model(
        FILE_PATH / ".." / "robots" / "planar_rrr" / "robot.urdf.xacro"
    )
    # Note that unlike rrr, planar_rrr's end_effector does not have a collision geometry
    assert set(adjacent_collisions(robot_model)) == set(
        [
            ("base_link", "link1", DisabledReason.ADJACENT),
            ("link1", "link2", DisabledReason.ADJACENT),
            ("link2", "link3", DisabledReason.ADJACENT),
        ]
    )
