import pathlib
from ramp.compute_disable_collisions import ComputeDisableCollisions


FILE_PATH = pathlib.Path(__file__).parent


def test_disable_collisions():
    cdc = ComputeDisableCollisions(
        FILE_PATH / ".." / "robots" / "rrr" / "rrr.urdf.xacro"
    )
    assert set(cdc.adjacents()) == set(
        [
            ("base_link", "link1"),
            ("link1", "link2"),
            ("link2", "link3"),
            ("link3", "end_effector"),
        ]
    )
    cdc = ComputeDisableCollisions(
        FILE_PATH
        / ".."
        / "external"
        / "mujoco_simulator"
        / "mujoco_simulator_py"
        / "examples"
        / "acrobot.xml"
    )
    assert set(cdc.adjacents()) == set([("upper_link", "lower_link")])

    cdc = ComputeDisableCollisions(
        FILE_PATH / ".." / "robots" / "planar_rrr" / "robot.urdf.xacro"
    )
    # Note that unlike rrr, planar_rrr's end_effector does not have a collision geometry
    assert set(cdc.adjacents()) == set(
        [
            ("base_link", "link1"),
            ("link1", "link2"),
            ("link2", "link3"),
        ]
    )
