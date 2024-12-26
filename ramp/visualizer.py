"""A visualizer for the robot using Meshcat."""

import meshcat
import meshcat_shapes
import pinocchio

from ramp.robot import Robot, RobotState


class Visualizer:
    """Meshcat visualizer for the robot."""

    def __init__(self, robot: Robot) -> None:
        """Initialize the visualizer.

        Args:
            robot: The robot to visualize.
        """
        self.robot = robot
        self.meshcat_visualizer = pinocchio.visualize.MeshcatVisualizer(
            robot.robot_model.model,
            robot.robot_model.collision_model,
            robot.robot_model.visual_model,
        )
        self.trajectory_visualizer = pinocchio.visualize.MeshcatVisualizer(
            robot.robot_model.model,
            robot.robot_model.collision_model,
            robot.robot_model.visual_model,
        )
        self.meshcat_visualizer.initViewer(
            viewer=meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000"),
        )
        self.trajectory_visualizer.initViewer(viewer=self.meshcat_visualizer.viewer)

        self.meshcat_visualizer.viewer.delete()
        self.trajectory_visualizer.viewer.delete()

        self.meshcat_visualizer.loadViewerModel()
        # TODO: The frames are pretty noisy - add a way to toggle them
        self.meshcat_visualizer.displayFrames(visibility=False)
        self.meshcat_visualizer.display(pinocchio.neutral(robot.robot_model.model))

        self.trajectory_visualizer.viewer["trajectory"].set_property(
            "visible",
            value=False,
        )
        self.trajectory_visualizer.loadViewerModel(rootNodeName="trajectory")
        self.trajectory_visualizer.display(pinocchio.neutral(robot.robot_model.model))

    def check_data(self, visualizer: pinocchio.visualize.MeshcatVisualizer):
        """Check if the model data changed and rebuild the data if needed."""
        if len(visualizer.visual_model.geometryObjects) != len(
            visualizer.visual_data.oMg,
        ):
            visualizer.rebuildData()
            visualizer.loadViewerModel(rootNodeName=visualizer.viewerRootNodeName)

    # TODO: Make it use q from pinocchio directly?
    def robot_state(self, robot_state: RobotState) -> None:
        """Visualize a robot state.

        Args:
            joint_positions: The joint positions to visualize.
        """
        self.check_data(self.meshcat_visualizer)
        self.meshcat_visualizer.display(robot_state.qpos)

    def robot_trajectory(self, waypoints: list[RobotState]) -> None:
        """Visualize a robot trajectory.

        Args:
            waypoints: The waypoints to visualize.
        """
        self.check_data(self.trajectory_visualizer)
        # TODO: No way to pass name to AnimationClip :(, the name for the animation is always `default`
        self.trajectory_visualizer.viewer["trajectory"].set_property(
            "visible",
            value=True,
        )
        animation = meshcat.animation.Animation()
        viewer = self.trajectory_visualizer.viewer
        for i, waypoint in enumerate(waypoints):
            with animation.at_frame(viewer, i) as frame:
                self.trajectory_visualizer.viewer = frame
                self.trajectory_visualizer.display(waypoint.qpos)
        self.trajectory_visualizer.viewer = viewer
        self.trajectory_visualizer.viewer.set_animation(animation)

    def frame(self, frame_name, transform):
        """Visualize a frame.

        Args:
            frame_name: The frame name to visualize.
            transform: The (4x4) transform of the frame.
        """
        meshcat_shapes.frame(self.meshcat_visualizer.viewer[frame_name])
        self.meshcat_visualizer.viewer[frame_name].set_transform(transform)
