import sys
import zenoh
from loop_rate_limiters import RateLimiter
import pathlib

from mujoco_simulator_py.mujoco_interface import MuJoCoInterface

FILE_PATH = pathlib.Path(__file__).parent


def main():
    if len(sys.argv) != 4:
        print("Usage: python attach_models_demo.py scene.xml MODEL.xml base_body_name")
        sys.exit(1)
    zenoh.init_log_from_env_or("error")
    mujoco_interface = MuJoCoInterface()
    mujoco_interface.reset(
        model_filename=FILE_PATH / sys.argv[1],
        keyframe="home",
    )
    for i, pos in enumerate(
        [[-0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [-0.5, -0.5, 0.0]]
    ):
        input("Press Enter...")
        mujoco_interface.attach_model(
            FILE_PATH / sys.argv[2],
            "world",
            sys.argv[3],
            (pos, [0.0, 0.0, 0.0, 1.0]),
            f"{i}/",
        )


if __name__ == "__main__":
    main()
