import pathlib
import sys
import numpy as np

import zenoh

from ramp import load_robot_model
from ramp.mujoco_interface import MuJoCoHardwareInterface

FILE_PATH = pathlib.Path(__file__).parent


def main():
    if len(sys.argv) != 4:
        print("Usage: python attach_models_demo.py scene.xml MODEL.xml base_body_name")
        sys.exit(1)
    zenoh.init_log_from_env_or("error")
    robot_model = load_robot_model(pathlib.Path(sys.argv[1]))
    mujoco_interface = MuJoCoHardwareInterface(robot_model)
    mujoco_interface.reset()
    nr = 5

    xv, yv = np.meshgrid(np.linspace(-2.5, 2.5, nr), np.linspace(-2.5, 2.5, nr))
    for i in range(nr):
        for j in range(nr):
            input("Press Enter...")
            mujoco_interface.attach_model(
                sys.argv[2],
                "world",
                sys.argv[3],
                ([xv[i, j], yv[i, j], 0.0], [0.0, 0.0, 0.0, 1.0]),
                f"{i}x{j}/",
            )


if __name__ == "__main__":
    main()
