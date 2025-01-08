import math
import sys
import time
import zenoh
from loop_rate_limiters import RateLimiter
import pathlib

from mujoco_simulator_py.mujoco_interface import MuJoCoInterface

FILE_PATH = pathlib.Path(__file__).parent


def main():
    if len(sys.argv) != 2:
        print("Usage: python attach_models_demo.py scene.xml")
        sys.exit(1)

    zenoh.init_log_from_env_or("error")
    mujoco_interface = MuJoCoInterface()
    mujoco_interface.reset(
        model_filename=FILE_PATH / sys.argv[1],
        keyframe="home",
    )
    # Zero in size means not needed
    objects = [
        ("sphere", "sphere", (0.0, 0.5, 0.5), (0.01, 0.0, 0.0)),
        ("capsule", "capsule", (0.1, 0.5, 0.5), (0.01, 0.02, 0.04)),
        ("ellipsoid", "ellipsoid", (0.2, 0.5, 0.5), (0.01, 0.02, 0.04)),
        ("cylinder", "cylinder", (0.3, 0.5, 0.5), (0.01, 0.02, 0.04)),
        ("box", "box", (0.4, 0.5, 0.5), (0.01, 0.01, 0.01)),
        ("arrow", "arrow", (0.5, 0.5, 0.5), (0.01, 0.01, 0.1)),
        ("arrow_single", "arrow1", (0.6, 0.5, 0.5), (0.01, 0.01, 0.1)),
        ("arrow_double", "arrow2", (0.7, 0.5, 0.5), (0.005, 0.005, 0.1)),
        ("line", "line", (0.8, 0.5, 0.5), (5.0, 0.0, 0.2)),
        ("linebox", "linebox", (0.9, 0.5, 0.5), (0.05, 0.05, 0.05)),
        ("triangle", "triangle", (1.0, 0.5, 0.5), (0.05, 0.05, 0.0)),
    ]
    for object_name, object_type, object_pos, object_size in objects:
        mujoco_interface.add_decorative_geometry(
            object_name, object_type, object_pos, object_size
        )
    time.sleep(2.0)
    for object_name, _, _, _ in objects:
        mujoco_interface.remove_decorative_geometry(object_name)

    for i in range(1000):
        mujoco_interface.add_decorative_geometry(
            "rotating_sphere",
            "sphere",
            (math.sin(i / 100), math.cos(i / 100), 0.0),
            (0.01, 0.0, 0.0),
        )
        time.sleep(0.01)


if __name__ == "__main__":
    main()
