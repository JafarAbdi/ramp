"""Example of using the differential_ik functionality."""

import logging
import pathlib
import time

from rich.logging import RichHandler

from ramp import load_robot_model, setup_logging
from ramp.ik_solver import IKSolver
from ramp.hardware_interface import MuJoCoHardwareInterface
from ramp.robot_model import create_geometry_object
import pinocchio
import hppfcl
import loop_rate_limiters
from ramp import Visualizer

setup_logging()
LOGGER = logging.getLogger(__name__)

GROUP_NAME = "arm"

robot_model = load_robot_model(pathlib.Path("robots/panda_mj/configs.toml"))
mj_interface = MuJoCoHardwareInterface(robot_model, "home")
visualizer = Visualizer(robot_model)

rate = loop_rate_limiters.RateLimiter(10)
rs = mj_interface.read()
rs.add_object(
    create_geometry_object(
        "sphere",
        hppfcl.Sphere(0.1),
        pinocchio.XYZQUATToSE3([0.5, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0]),
    ),
)

while True:
    # TODO: Need a better way to do this!!
    rs.qpos = mj_interface.read().qpos
    distance = rs.compute_distance("hand_0", "sphere")
    visualizer.point("p1", distance.getNearestPoint1())
    visualizer.point("p2", distance.getNearestPoint2())
    visualizer.robot_state(rs)
    rate.sleep()
