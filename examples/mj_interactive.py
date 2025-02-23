"""Example of using the differential_ik functionality."""

import logging
import pathlib
import time
import sys

from rich.logging import RichHandler

from ramp import load_robot_model, setup_logging
from ramp.ik_solver import IKSolver
from ramp.mujoco_interface import MuJoCoHardwareInterface
from ramp.robot_model import create_geometry_object
import pinocchio
import hppfcl
import loop_rate_limiters
from mujoco_simulator_msgs.mujoco_simulator_pb2 import GeometryType

from ramp import Visualizer

setup_logging()
LOGGER = logging.getLogger(__name__)

GROUP_NAME = "arm"

robot_model = load_robot_model(pathlib.Path("robots/panda_mj/configs.toml"))
mj_interface = MuJoCoHardwareInterface(robot_model, "home")

rs = mj_interface.read()
rs.add_object(
    create_geometry_object(
        "sphere",
        hppfcl.Sphere(0.1),
        pinocchio.XYZQUATToSE3([0.5, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0]),
    ),
)
mj_interface.add_decorative_geometry(
    "collision_sphere", GeometryType.SPHERE, (0.5, 0.0, 0.25), (0.1, 0.0, 0.0)
)

visualizer = None
if len(sys.argv) > 1 and sys.argv[1] == "visualize":
    visualizer = Visualizer(robot_model)
    visualizer.robot_state(mj_interface.read())


rate = loop_rate_limiters.RateLimiter(10)
while True:
    rs = mj_interface.read()
    distance = rs.compute_distance("hand_0", "sphere")
    mj_interface.add_decorative_geometry(
        "p1", GeometryType.SPHERE, list(distance.getNearestPoint1()), (0.01, 0.0, 0.0)
    )
    mj_interface.add_decorative_geometry(
        "p2", GeometryType.SPHERE, list(distance.getNearestPoint2()), (0.01, 0.0, 0.0)
    )
    if visualizer is not None:
        visualizer.point("p1", distance.getNearestPoint1())
        visualizer.point("p2", distance.getNearestPoint2())
        visualizer.robot_state(rs)
    rate.sleep()
