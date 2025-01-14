"""Motion planning module using OMPL."""

import functools
import logging
import os
import platform
import re
from collections.abc import Callable

import numpy as np
import ompl
from ompl import base as ob
from ompl import geometric as og

from ramp.constants import (
    PINOCCHIO_FREEFLYER_JOINT,
    PINOCCHIO_PLANAR_JOINT,
    PINOCCHIO_PRISMATIC_JOINT,
    PINOCCHIO_REVOLUTE_JOINT,
    PINOCCHIO_REVOLUTE_UNALIGNED_JOINT,
    PINOCCHIO_SPHERICAL_JOINT,
    PINOCCHIO_TRANSLATION_JOINT,
    PINOCCHIO_UNBOUNDED_JOINT,
)
from ramp.pinocchio_utils import joint_nq
from ramp.robot_model import RobotModel
from ramp.robot_state import RobotState

LOGGER = logging.getLogger(__name__)


def get_ompl_log_level(level: str) -> ompl.util.LogLevel:
    """Get OMPL log level.

    Args:
        level: Log level

    Returns:
        OMPL log level
    """
    level = level.upper()
    # Why using ompl.util.LOG_DEBUG doesn't work on MacOS?
    logger_module = ompl.base if platform.system() == "Darwin" else ompl.util
    if level == "DEBUG":
        return logger_module.LOG_DEBUG
    if level == "INFO":
        return logger_module.LOG_INFO
    if level == "WARN":
        return logger_module.LOG_WARN
    if level == "ERROR":
        return logger_module.LOG_ERROR
    msg = f"Unknown log level: {level}"
    raise ValueError(msg)


ompl.util.setLogLevel(get_ompl_log_level(os.getenv("LOG_LEVEL", "ERROR")))


def list2vec(input_list):
    """Convert a list to an OMPL vector."""
    ret = ompl.util.vectorDouble()
    for e in input_list:
        ret.append(e)
    return ret


def get_ompl_planners() -> list[str]:
    """Get OMPL planners.

    Returns:
        List of OMPL planners.
    """
    from inspect import isclass

    module = ompl.geometric
    planners = []
    for obj in dir(module):
        planner_name = f"{module.__name__}.{obj}"
        planner = eval(planner_name)  # noqa: S307
        if isclass(planner) and issubclass(planner, ompl.base.Planner):
            planners.append(
                planner_name.split("ompl.geometric.")[1],
            )  # Name is ompl.geometric.<planner>
    return planners


# Actually the type is CompoundStateInternal
def from_ompl_state(space: ob.CompoundStateSpace, state: ob.State) -> list[float]:
    """Convert ompl state to joint positions."""
    assert isinstance(space, ob.CompoundStateSpace)
    assert isinstance(state, ob.CompoundStateInternal)
    joint_positions = []
    for space_idx in range(space.getSubspaceCount()):
        subspace = space.getSubspace(space_idx)
        substate = state[space_idx]
        match subspace.getType():
            case ob.STATE_SPACE_REAL_VECTOR:
                joint_positions.extend(
                    [substate[i] for i in range(subspace.getDimension())],
                )
            case ob.STATE_SPACE_SO3:
                joint_positions.extend([substate.x, substate.y, substate.z, substate.w])
            case (
                ob.STATE_SPACE_SE2
                | ob.STATE_SPACE_DUBINS
                | ob.STATE_SPACE_REEDS_SHEPP
            ):
                joint_positions.extend(
                    [substate.getX(), substate.getY(), substate.getYaw()],
                )
            case ob.STATE_SPACE_SO2:
                joint_positions.append(substate.value)
            case ob.STATE_SPACE_SE3:
                rotation = substate.rotation()
                joint_positions.extend(
                    [
                        substate.getX(),
                        substate.getY(),
                        substate.getZ(),
                        rotation.x,
                        rotation.y,
                        rotation.z,
                        rotation.w,
                    ],
                )
            case _:
                msg = f"Unsupported space: {subspace}"
                raise ValueError(msg)
    return joint_positions


class PathClearanceObjective(ob.StateCostIntegralObjective):
    """Path clearance objective."""

    def __init__(
        self,
        si: ob.SpaceInformation,
        robot_state: RobotState,
        group_name: str,
    ) -> None:
        """Initialize the path clearance objective.

        Args:
            si: The space information.
            robot_state: The reference robot state.
            group_name: The group name.
        """
        super(PathClearanceObjective, self).__init__(  # noqa: UP008
            si,
            enableMotionCostInterpolation=False,
        )
        self.si_ = si
        self.robot_state = robot_state.clone()
        self._group_name = group_name

    def stateCost(self, s):  # noqa: N802
        """Compute the cost of a state."""
        self.robot_state[self._group_name] = from_ompl_state(
            self.si_.getStateSpace(),
            s,
        )
        return ob.Cost(
            1.0
            / (
                np.sum(
                    self.robot_state.compute_distances(
                        "mobile_base_0",
                        "sphere_0x0x0",
                    ),  # TODO: Make it as a lot
                )
                + 0.1
            ),
        )


# MoveIt has ProjectionEvaluatorLinkPose/ProjectionEvaluatorJointValue
class ProjectionEvaluatorLinkPose(ob.ProjectionEvaluator):
    """OMPL projection evaluator."""

    def __init__(self, space, pose_fn: Callable[[ob.State], np.ndarray]) -> None:
        """Initialize the projection evaluator.

        Args:
            space: The state space
            pose_fn: The pose function to project a state to a link's position
        """
        super().__init__(space)
        self._pose_fn = pose_fn
        self.defaultCellSizes()

    def getDimension(self):  # noqa: N802
        """Get the dimension of the projection."""
        return 3

    def defaultCellSizes(self):  # noqa: N802
        """Set the default cell sizes."""
        # TODO: Should use ompl.tools.PROJECTION_DIMENSION_SPLITS
        # space.getBounds() -> bounds_.getDifference() see ompl's repo
        self.cellSizes_ = list2vec([0.1, 0.1, 0.1])

    def project(self, state, projection):
        """Project the input state.

        Args:
            state: The input state
            projection: The projected state
        """
        pose = self._pose_fn(state)
        projection[0] = pose[0, 3]
        projection[1] = pose[1, 3]
        projection[2] = pose[2, 3]


class MotionPlanner:
    """A wrapper for OMPL planners."""

    def __init__(self, robot_model: RobotModel, group_name: str) -> None:
        """Initialize the motion planner.

        Args:
            robot_model: The robot model.
            group_name: The group to plan for.
        """
        self._robot_model = robot_model
        self._group_name = group_name
        self._space = ob.CompoundStateSpace()
        for idx in robot_model[group_name].joint_indices:
            joint = robot_model.model.joints[int(idx)]
            joint_name = robot_model.model.names[int(idx)]
            joint_type = joint.shortname()
            if (
                re.match(PINOCCHIO_REVOLUTE_JOINT, joint_type)
                or (re.match(PINOCCHIO_PRISMATIC_JOINT, joint_type))
                or joint_type == PINOCCHIO_REVOLUTE_UNALIGNED_JOINT
            ):
                bounds = ob.RealVectorBounds(1)
                space = ob.RealVectorStateSpace(1)
                bounds.setLow(
                    0,
                    robot_model.model.lowerPositionLimit[joint.idx_q],
                )
                bounds.setHigh(
                    0,
                    robot_model.model.upperPositionLimit[joint.idx_q],
                )
                space.setBounds(bounds)
                self._space.addSubspace(space, 1.0)
            elif re.match(PINOCCHIO_UNBOUNDED_JOINT, joint_type):
                self._space.addSubspace(
                    ob.SO2StateSpace(),
                    1.0,
                )
            elif joint_type == PINOCCHIO_PLANAR_JOINT:
                bounds = ob.RealVectorBounds(2)
                bounds.setLow(0, -10.0)  # robot.model.lowerPositionLimit[joint.idx_q])
                bounds.setHigh(0, 10.0)  # robot.model.upperPositionLimit[joint.idx_q])
                bounds.setLow(
                    1,
                    -10.0,
                )  # robot.model.lowerPositionLimit[joint.idx_q + 1])
                bounds.setHigh(
                    1,
                    10.0,
                )  # robot.model.upperPositionLimit[joint.idx_q + 1])
                if robot_model.motion_model.get(joint_name) == "dubins":
                    space = ob.DubinsStateSpace(
                        1.0,  # Turning radius
                        isSymmetric=True,  # If this is false, it's struggling to find a solution
                    )
                    space.setBounds(bounds)
                    self._space.addSubspace(space, 1.0)
                elif robot_model.motion_model.get(joint_name) == "reeds_shepp":
                    space = ob.ReedsSheppStateSpace(1.0)  # turning radius
                    space.setBounds(bounds)
                    self._space.addSubspace(space, 1.0)
                else:
                    space = ob.SE2StateSpace()
                    space.setBounds(bounds)
                    self._space.addSubspace(space, 1.0)
            elif joint_type == PINOCCHIO_SPHERICAL_JOINT:
                self._space.addSubspace(ob.SO3StateSpace(), 1.0)
            elif joint_type == PINOCCHIO_TRANSLATION_JOINT:
                self._space.addSubspace(ob.RealVectorStateSpace(3), 1.0)
            elif joint_type == PINOCCHIO_FREEFLYER_JOINT:
                # TODO: Parameterize -10/10
                bounds = ob.RealVectorBounds(3)
                bounds.setLow(-10)
                bounds.setHigh(10)
                space = ob.SE3StateSpace()
                space.setBounds(bounds)
                self._space.addSubspace(space, 1.0)
            else:
                msg = f"Unknown joint type: '{joint_type}' for joint '{robot_model.model.names[int(idx)]}'"
                raise ValueError(msg)

        self._setup = og.SimpleSetup(self._space)

    def _get_planner(self, planner):
        try:
            return eval(  # noqa: S307
                f"og.{planner}(self._setup.getSpaceInformation())",
            )
        except AttributeError:
            LOGGER.exception(
                f"Planner '{planner}' not found - Available planners: {get_ompl_planners()}",
            )
            raise

    def as_ompl_state(self, robot_state: RobotState) -> ob.State:
        """Convert joint positions to ompl state."""
        state = ob.CompoundState(self._space)
        internal_state = state()
        i = 0
        joint_positions = robot_state.group_qpos(self._group_name)
        for space_idx, space in enumerate(self._space.getSubspaces()):
            # TODO: No need to have this, we can get the space dimension from state
            joint_index = self._robot_model[self._group_name].joint_indices[space_idx]
            size = joint_nq(self._robot_model.model.joints[int(joint_index)])
            match space.getType():
                case ob.STATE_SPACE_SO2:
                    substate = internal_state[space_idx]
                    substate.value = joint_positions[i]
                    i += size
                case (
                    ob.STATE_SPACE_SE2
                    | ob.STATE_SPACE_DUBINS
                    | ob.STATE_SPACE_REEDS_SHEPP
                ):
                    substate = internal_state[space_idx]
                    substate.setX(joint_positions[i])
                    substate.setY(joint_positions[i + 1])
                    substate.setYaw(joint_positions[i + 2])
                    i += size
                case ob.STATE_SPACE_SO3:
                    substate = internal_state[space_idx]
                    substate.x = joint_positions[i]
                    substate.y = joint_positions[i + 1]
                    substate.z = joint_positions[i + 2]
                    substate.w = joint_positions[i + 3]
                    i += size
                case ob.STATE_SPACE_REAL_VECTOR:
                    substate = internal_state[space_idx]
                    assert size == self._space.getSubspace(space_idx).getDimension()
                    for idx in range(size):
                        substate[idx] = joint_positions[i + idx]
                    i += size
                case ob.STATE_SPACE_SE3:
                    substate = internal_state[space_idx]
                    substate.setX(joint_positions[i])
                    substate.setY(joint_positions[i + 1])
                    substate.setZ(joint_positions[i + 2])
                    rotation = substate.rotation()
                    rotation.x = joint_positions[i + 3]
                    rotation.y = joint_positions[i + 4]
                    rotation.z = joint_positions[i + 5]
                    rotation.w = joint_positions[i + 6]
                    i += size
                case _:
                    msg = f"Unsupported space: {space}"
                    raise ValueError(msg)
        assert i == len(joint_positions)
        return ob.State(state)

    def _setup_state_validity_checker(self, reference_state: RobotState):
        """Set the state validity checker."""

        def is_ompl_state_valid(reference_robot_state, state):
            reference_robot_state.set_group_qpos(
                self._group_name,
                from_ompl_state(
                    self._space,
                    state,
                ),
            )
            return self.is_state_valid(reference_robot_state)

        self._setup.setStateValidityChecker(
            ob.StateValidityCheckerFn(
                functools.partial(is_ompl_state_valid, reference_state.clone()),
            ),
        )

    def _setup_projection_evaluator(self, reference_state: RobotState):
        """Set the projection evaluator."""
        if self._robot_model[self._group_name].tcp_link_name is not None:

            def pose_fn(reference_robot_state, state):
                reference_robot_state.set_group_qpos(
                    self._group_name,
                    from_ompl_state(
                        self._space,
                        state,
                    ),
                )
                return reference_robot_state.get_frame_pose(
                    self._robot_model[self._group_name].tcp_link_name,
                ).np

            self._space.registerDefaultProjection(
                ProjectionEvaluatorLinkPose(
                    self._space,
                    functools.partial(pose_fn, reference_state.clone()),
                ),
            )

    # TODO: Add termination conditions doc/markdown/plannerTerminationConditions.md
    def plan(
        self,
        start_state: RobotState,
        goal_state: RobotState,
        timeout: float = 1.0,
        planner: str | None = None,
    ) -> list[RobotState] | None:
        """Plan a trajectory from start to goal.

        Args:
            start_state: The start robot state.
            goal_state: The goal robot state.
            timeout: Timeout for planner
            planner: The planner to use.

        Returns:
            The trajectory as a list of joint positions or None if no solution was found.
        """
        # Use start state as reference state for state validity checker and projection evaluator
        group_goal_qpos = goal_state.group_qpos(self._group_name)
        goal_state = start_state.clone()
        goal_state.set_group_qpos(self._group_name, group_goal_qpos)

        self._setup.clear()
        self._setup_state_validity_checker(start_state)
        self._setup_projection_evaluator(start_state)

        LOGGER.debug(self._setup.getStateSpace().settings())

        if not self.is_state_valid(start_state, verbose=True):
            LOGGER.error("Start state is invalid - in collision or out of bounds")
            return None
        if not self.is_state_valid(goal_state, verbose=True):
            LOGGER.error("Goal state is invalid - in collision or out of bounds")
            return None
        self._setup.setStartAndGoalStates(
            self.as_ompl_state(start_state),
            self.as_ompl_state(goal_state),
        )

        if planner is not None:
            self._setup.setPlanner(self._get_planner(planner))

        solve_result = self._setup.solve(timeout)
        if not solve_result:
            LOGGER.info("Did not find solution!")
            return None
        path = self._setup.getSolutionPath()
        if not path.check():
            LOGGER.warning("Path fails check!")

        if (
            ob.PlannerStatus.getStatus(solve_result)
            == ob.PlannerStatus.APPROXIMATE_SOLUTION
        ):
            LOGGER.warning("Found approximate solution!")

        LOGGER.debug("Simplifying solution..")
        LOGGER.debug(
            f"Path length before simplification: {path.length()} with {len(path.getStates())} states",
        )
        self._setup.simplifySolution()
        simplified_path = self._setup.getSolutionPath()
        LOGGER.debug(
            f"Path length after simplifySolution: {simplified_path.length()} with {len(simplified_path.getStates())} states",
        )
        # self._setup.getPathSimplifier() Fails with
        # TypeError: No Python class registered for C++ class std::shared_ptr<ompl::geometric::PathSimplifier>
        path_simplifier = og.PathSimplifier(self._setup.getSpaceInformation())
        path_simplifier.ropeShortcutPath(simplified_path)
        LOGGER.debug(
            f"Simplified path length after ropeShortcutPath: {simplified_path.length()} with {len(simplified_path.getStates())} states",
        )
        path_simplifier.smoothBSpline(simplified_path)
        LOGGER.debug(
            f"Simplified path length after smoothBSpline: {simplified_path.length()} with {len(simplified_path.getStates())} states",
        )

        if not simplified_path.check():
            LOGGER.warning("Simplified path fails check!")

        LOGGER.debug("Interpolating simplified path...")
        simplified_path.interpolate()

        if not simplified_path.check():
            LOGGER.warning("Interpolated simplified path fails check!")

        solution = []
        reference_robot_state = start_state.clone()
        for state in simplified_path.getStates():
            reference_robot_state.set_group_qpos(
                self._group_name,
                from_ompl_state(
                    self._space,
                    state,
                ),
            )
            solution.append(reference_robot_state.clone())
        LOGGER.info(f"Found solution with {len(solution)} waypoints")
        return solution

    def is_state_valid(self, robot_state: RobotState, *, verbose=False) -> bool:
        """Check if the state is valid, i.e. not in collision or out of bounds.

        Args:
            robot_state: The robot state to check.
            verbose: Whether to log additional information.

        Returns:
            True if the state is valid, False otherwise.
        """
        ompl_state = self.as_ompl_state(robot_state)
        return self._setup.getSpaceInformation().satisfiesBounds(
            ompl_state(),
        ) and not robot_state.check_collision(verbose=verbose)
