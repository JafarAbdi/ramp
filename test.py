"""Retime an one dimensional path
===============================
"""

import json
import time

import zmq


class ZMQPublisher:
    def __init__(self, zmq_url):
        """Initialize ZMQ Publisher
        Args:
            zmq_url (str): ZMQ server URL (e.g., "tcp://*:5555")
        """
        self.zmq_url = zmq_url
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(self.zmq_url)
        print(f"Bound to ZMQ server at {self.zmq_url}")
        time.sleep(1.0)  # Allow time for ZMQ connection to stabilize

    def publish(
        self,
        data: dict[str, float],
        timestamp: None | float = None,
        topic="data",
    ):
        """Publish data to ZMQ server
        Args:
            data: Data to publish
            topic (str): Topic for the message
        """
        try:
            message = json.dumps(
                data
                | {"timestamp": timestamp if timestamp is not None else time.time()},
            )

            # Based on https://github.com/facontidavide/PlotJuggler/blob/main/plotjuggler_plugins/DataStreamZMQ/utilities/start_test_publisher.py
            packet = [
                topic.encode("utf-8"),
                message.encode("utf-8"),
            ]
            self.socket.send_multipart(packet)

        except Exception as e:
            print(f"Failed to publish data: {e}")
            raise

    def close(self):
        """Close the connection and clean up resources"""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        print("ZMQ Publisher closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


################################################################################
# Import necessary libraries.
import numpy as np
import toppra as ta
import toppra.algorithm as algo
from toppra import constraint

ta.setup_logging("INFO")

################################################################################
# We now generate a simply path.  When constructing a path, you must
# "align" the waypoint properly yourself. For instance, if the
# waypoints are [0, 1, 10] like in the above example, the path
# position should be aligned like [0, 0.1, 1.0]. If this is not done,
# the CubicSpline Interpolator might result undesirable oscillating
# paths!

waypts = [[0], [1], [10]]
path = ta.SplineInterpolator([0, 0.1, 1.0], waypts)


################################################################################
# Setup the velocity and acceleration
vlim = np.array([[-3, 3]])
alim = np.array([[-4, 4]])
pc_vel = constraint.JointVelocityConstraint(vlim)
pc_acc = constraint.JointAccelerationConstraint(
    alim,
    discretization_scheme=constraint.DiscretizationType.Interpolation,
)


################################################################################
# Setup the problem instance and solve it.
instance = algo.TOPPRA([pc_vel, pc_acc], path, solver_wrapper="seidel")
jnt_traj = instance.compute_trajectory(0, 0)

################################################################################
zmq_publisher = ZMQPublisher("tcp://*:5555")
# We can now visualize the result
duration = jnt_traj.duration
print(f"Found optimal trajectory with duration {duration:f} sec")
ts = np.linspace(0, duration, 100)
for t in ts:
    qs = jnt_traj.eval(t)
    qds = jnt_traj.evald(t)
    qdds = jnt_traj.evaldd(t)
    zmq_publisher.publish(
        {
            "position": qs.item(),
            "velocity": qds.item(),
            "acceleration": qdds.item(),
        },
        timestamp=t.item(),
    )
