import math
import pathlib
import sys
import time

import zenoh

from ramp.mujoco_interface import MuJoCoHardwareInterface
from ramp import load_robot_model
from mujoco_simulator_msgs.mujoco_simulator_pb2 import GeometryType

FILE_PATH = pathlib.Path(__file__).parent


def main():
    if len(sys.argv) != 2:
        print("Usage: python decorative_geometries.py scene.xml")
        sys.exit(1)

    zenoh.init_log_from_env_or("error")
    robot_model = load_robot_model(FILE_PATH / sys.argv[1])
    mujoco_interface = MuJoCoHardwareInterface(robot_model)
    mujoco_interface.reset(keyframe="home")
    # Zero in size means not needed
    objects = [
        ("sphere", GeometryType.SPHERE, (0.0, 0.5, 0.5), (0.01, 0.0, 0.0)),
        ("capsule", GeometryType.CAPSULE, (0.1, 0.5, 0.5), (0.01, 0.02, 0.04)),
        ("ellipsoid", GeometryType.ELLIPSOID, (0.2, 0.5, 0.5), (0.01, 0.02, 0.04)),
        ("cylinder", GeometryType.CYLINDER, (0.3, 0.5, 0.5), (0.01, 0.02, 0.04)),
        ("box", GeometryType.BOX, (0.4, 0.5, 0.5), (0.01, 0.01, 0.01)),
        ("arrow", GeometryType.ARROW, (0.5, 0.5, 0.5), (0.01, 0.01, 0.1)),
        ("arrow_single", GeometryType.ARROW_SINGLE, (0.6, 0.5, 0.5), (0.01, 0.01, 0.1)),
        (
            "arrow_double",
            GeometryType.ARROW_DOUBLE,
            (0.7, 0.5, 0.5),
            (0.005, 0.005, 0.1),
        ),
        ("line", GeometryType.LINE, (0.8, 0.5, 0.5), (5.0, 0.0, 0.2)),
        ("linebox", GeometryType.LINEBOX, (0.9, 0.5, 0.5), (0.05, 0.05, 0.05)),
        ("triangle", GeometryType.TRIANGLE, (1.0, 0.5, 0.5), (0.05, 0.05, 0.0)),
    ]
    for object_name, object_type, object_pos, object_size in objects:
        mujoco_interface.add_decorative_geometry(
            object_name,
            object_type,
            object_pos,
            object_size,
        )
    time.sleep(2.0)
    for object_name, _, _, _ in objects:
        mujoco_interface.remove_decorative_geometry(object_name)

    for i in range(1000):
        mujoco_interface.add_decorative_geometry(
            "rotating_sphere",
            GeometryType.SPHERE,
            (math.sin(i / 100), math.cos(i / 100), 0.0),
            (0.01, 0.0, 0.0),
        )
        time.sleep(0.01)


if __name__ == "__main__":
    main()
