"""CasADiRobot class for representing a robot in CasADi."""

import casadi
import pinocchio
import pinocchio.visualize
from pinocchio import casadi as cpin

from ramp.robot_model import RobotModel


# TODO: Maybe delete and combine with Robot class?
# Prefix with c for CasADi
# Example: cmodel, cdata, cq, cjacobian
class CasADiRobot:
    """A class to represent the robot in CasADi."""

    def __init__(self, robot_model: RobotModel):
        """Initialize the CasADi robot.

        Args:
            robot_model: The robot model to use for the CasADi robot
        """
        self.model = cpin.Model(robot_model.model)
        self.data = self.model.createData()
        self.q = casadi.SX.sym("q", robot_model.model.nq, 1)
        cpin.framesForwardKinematics(self.model, self.data, self.q)
        cpin.updateFramePlacements(self.model, self.data)

    def jacobian(
        self,
        target_frame_name,
        reference_frame=pinocchio.ReferenceFrame.LOCAL,
    ):
        """Calculate the Jacobian of a frame.

        Args:
            target_frame_name: The target frame name
            reference_frame: The reference frame

        Returns:
            The Jacobian matrix of shape (6, n) where n is the number of joints.
        """
        return cpin.computeFrameJacobian(
            self.model,
            self.data,
            self.q,
            self.model.getFrameId(target_frame_name),
            reference_frame,
        )
