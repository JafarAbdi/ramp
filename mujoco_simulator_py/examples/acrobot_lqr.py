import numpy as np
import scipy
import zenoh
from loop_rate_limiters import RateLimiter
import pathlib

from mujoco_simulator_py.mujoco_interface import MuJoCoInterface

FILE_PATH = pathlib.Path(__file__).parent


class LQR:
    def __init__(self):
        # Linearised dynamics around the upright equilibrium
        self.A = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [9.558, -12.467, 0.0, 0.0],
                [-14.684, 44.744, 0.0, 0.0],
            ],
        )
        self.B = np.array([[0.0], [0.0], [-2.692], [8.774]])
        Q = np.diag((1.0, 1.0, 1.0, 1.0))
        R = np.array([[0.1]])
        self.P = scipy.linalg.solve_continuous_are(self.A, self.B, Q, R)
        # Compute the feedback gain matrix K.
        self.K = np.linalg.inv(R) @ self.B.T @ self.P
        self.ctrl_0 = 0.0  # Linearisation control point
        self.qpos_0 = np.array([3.14, 0.0])  # Linearisation state point

    def control(self, dx):
        return self.ctrl_0 - self.K @ dx


def main():
    zenoh.init_log_from_env_or("error")
    mujoco_interface = MuJoCoInterface()
    mujoco_interface.reset(
        model_filename=FILE_PATH / "acrobot.xml",
        keyframe="upright",
    )

    lqr = LQR()
    rate = RateLimiter(250)
    while True:
        qpos = mujoco_interface.qpos()
        qvel = mujoco_interface.qvel()
        dx = np.hstack(
            (qpos - lqr.qpos_0, qvel),
        ).T
        mujoco_interface.ctrl({0: lqr.control(dx).item()})
        rate.sleep()


if __name__ == "__main__":
    main()
