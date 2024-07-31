"""A wrapper for the TRAC-IK solver."""

import trac_ik_py


class IKSolver:
    """A wrapper for the TRAC-IK solver."""

    def __init__(self, filename, base_link_name, tcp_link_name) -> None:
        """Initialize the IK solver.

        Args:
            filename: The MJCF/URDF filename.
            base_link_name: The base link name.
            tcp_link_name: The TCP link name.
        """
        self._ik_solver = trac_ik_py.TRAC_IK(
            base_link_name,
            tcp_link_name,
            str(filename),
            0.005,  # timeout
            1e-5,  # epsilon
            trac_ik_py.SolveType.Speed,
        )

    def tcp_pose(self, joint_positions):
        """Get the pose of the tcp_link as [x, y, z, rx, ry, rz, rw].

        Args:
            joint_positions: Input joint positions
        """
        assert len(joint_positions) == self._ik_solver.getNrOfJointsInChain()
        # JntToCart returns [x, y, z, rw, rx, ry, rz]
        pose = self._ik_solver.JntToCart(joint_positions)
        return [pose[0], pose[1], pose[2], pose[4], pose[5], pose[6], pose[3]]

    def solve(self, target_pose, seed=None):
        """Compute the inverse kinematics of the robot for a given target pose.

        Args:
            target_pose: The target pose [x, y, z, qx, qy, qz, qw] of the tcp_link_name w.r.t. the base_link_name
            seed: IK Solver seed

        Returns:
            The joint positions for the target pose or None if no solution was found
        """
        assert len(seed) == self._ik_solver.getNrOfJointsInChain(), (
            f"Seed has {len(seed)} joints, expected {self._ik_solver.getNrOfJointsInChain()}"
            f" - joint names: {self._ik_solver.getJointNamesInChain()}"
            f" - joint bounds: ({self._ik_solver.getLowerBoundLimits()}, {self._ik_solver.getUpperBoundLimits()})"
        )
        target_pose = [
            target_pose[0],
            target_pose[1],
            target_pose[2],
            target_pose[6],
            target_pose[3],
            target_pose[4],
            target_pose[5],
        ]
        joint_positions = self._ik_solver.CartToJnt(
            seed,
            target_pose,
            [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3],
        )
        if len(joint_positions) == 0:
            return None
        return joint_positions
