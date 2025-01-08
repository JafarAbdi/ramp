from ramp.mujoco_interface import MuJoCoHardwareInterface
from ramp import load_robot_model
import pathlib
import pytest

FILE_PATH = pathlib.Path(__file__).parent


@pytest.mark.skip("This test is not working")
def test_mujoco_interface():
    robot_model = load_robot_model(
        FILE_PATH / ".." / "robots" / "acrobot" / "acrobot.xml"
    )
    mj_interface = MuJoCoHardwareInterface(robot_model)
