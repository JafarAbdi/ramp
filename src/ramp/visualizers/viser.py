# Copied from https://github.com/stack-of-tasks/pinocchio/blob/devel/bindings/python/pinocchio/visualize/viser_visualizer.py until it get released
import time

import hppfcl
import numpy as np
import pinocchio as pin
import trimesh  # Required by viser
import viser
from pinocchio.visualize.base_visualizer import BaseVisualizer

from ramp.robot_model import RobotModel
from ramp.robot_state import RobotState

MESH_TYPES = (hppfcl.BVHModelBase, hppfcl.HeightFieldOBBRSS, hppfcl.HeightFieldAABB)


def check_data(visualizer: pin.visualize.BaseVisualizer):
    """Check if the model data changed and rebuild the data if needed."""
    if not visualizer.model.check(visualizer.data):
        visualizer.rebuildData()
        visualizer.loadViewerModel(rootNodeName=visualizer.viewerRootNodeName)


class PinocchioViserVisualizer(BaseVisualizer):
    """A Pinocchio visualizer using Viser."""

    def __init__(
        self,
        model=pin.Model(),
        collision_model=None,
        visual_model=None,
        copy_models=False,
        data=None,
        collision_data=None,
        visual_data=None,
    ):
        super().__init__(
            model,
            collision_model,
            visual_model,
            copy_models,
            data,
            collision_data,
            visual_data,
        )
        self.static_objects = []

    def initViewer(
        self,
        viewer=None,
        open=False,
        loadModel=False,
        host="localhost",
        port="8000",
    ):
        """Start a new Viser server and client.
        Note: the server can also be passed in through the `viewer` argument.

        Parameters:
            viewer: An existing ViserServer instance, or None.
                If None, creates a new ViserServer in this visualizer.
            open: If True, automatically opens a browser tab to the visualizer.
            loadModel: If True, loads the Pinocchio models passed in.
            host: If `viewer` is None, this will be the host URL.
            port: If `viewer` is None, this will be the host port.
        """
        if (viewer is not None) and (not isinstance(viewer, viser.ViserServer)):
            raise RuntimeError(
                "'viewer' argument must be None or a valid ViserServer instance.",
            )

        self.viewer = viewer or viser.ViserServer(host=host, server_port=port)
        self.frames = {}

        if open:
            import webbrowser

            webbrowser.open(f"http://{self.viewer.get_host()}:{self.viewer.get_port()}")

            # Wait until clients are reported.
            # Otherwise, capturing an image too soon after opening a browser window
            # may not register any clients.
            while len(self.viewer.get_clients()) == 0:
                time.sleep(0.1)

        if loadModel:
            self.loadViewerModel()

    def loadViewerModel(
        self,
        rootNodeName="pinocchio",
        collision_color=None,
        visual_color=None,
        frame_axis_length=0.2,
        frame_axis_radius=0.01,
    ):
        """Load the robot in a Viser viewer.

        Parameters:
            rootNodeName: name to give to the robot in the viewer
            collision_color: optional, color to give to the collision model of
                the robot. Format is a list of four RGBA floating-point numbers
                (between 0 and 1)
            visual_color: optional, color to give to the visual model of
                the robot. Format is a list of four RGBA floating-point numbers
                (between 0 and 1)
            frame_axis_length: optional, length of frame axes if displaying frames.
            frame_axis_radius: optional, radius of frame axes if displaying frames.
        """
        self.viewerRootNodeName = rootNodeName

        # Create root frames to help toggle visibility in the Viser UI.
        self.visualRootNodeName = rootNodeName + "/visual"
        self.visualRootFrame = self.viewer.scene.add_frame(
            self.visualRootNodeName,
            show_axes=False,
        )
        self.collisionRootNodeName = rootNodeName + "/collision"
        self.collisionRootFrame = self.viewer.scene.add_frame(
            self.collisionRootNodeName,
            show_axes=False,
        )
        self.framesRootNodeName = rootNodeName + "/frames"
        self.framesRootFrame = self.viewer.scene.add_frame(
            self.framesRootNodeName,
            show_axes=False,
        )

        # Load visual model
        if (visual_color is not None) and (len(visual_color) != 4):
            raise RuntimeError("visual_color must have 4 elements for RGBA.")
        if self.visual_model is not None:
            for visual in self.visual_model.geometryObjects:
                self.loadViewerGeometryObject(
                    visual,
                    self.visualRootNodeName,
                    visual_color,
                )
        self.displayVisuals(True)

        # Load collision model
        if (collision_color is not None) and (len(collision_color) != 4):
            raise RuntimeError("collision_color must have 4 elements for RGBA.")
        if self.collision_model is not None:
            for collision in self.collision_model.geometryObjects:
                self.loadViewerGeometryObject(
                    collision,
                    self.collisionRootNodeName,
                    collision_color,
                )
        self.displayCollisions(False)

        # Load frames
        for frame in self.model.frames:
            frame_name = self.framesRootNodeName + "/" + frame.name
            self.frames[frame_name] = self.viewer.scene.add_frame(
                frame_name,
                show_axes=True,
                axes_length=frame_axis_length,
                axes_radius=frame_axis_radius,
            )
        self.displayFrames(False)

    def loadViewerGeometryObject(self, geometry_object, prefix="", color=None):
        """Loads a single geometry object."""
        name = geometry_object.name
        if prefix:
            name = prefix + "/" + name
        geom = geometry_object.geometry
        color_override = color or geometry_object.meshColor

        if isinstance(geom, hppfcl.Box):
            frame = self.viewer.scene.add_box(
                name,
                dimensions=geom.halfSide * 2.0,
                color=color_override[:3],
                opacity=color_override[3],
            )
        elif isinstance(geom, hppfcl.Sphere):
            frame = self.viewer.scene.add_icosphere(
                name,
                radius=geom.radius,
                color=color_override[:3],
                opacity=color_override[3],
            )
        elif isinstance(geom, hppfcl.Cylinder):
            mesh = trimesh.creation.cylinder(
                radius=geom.radius,
                height=geom.halfLength * 2.0,
            )
            frame = self.viewer.scene.add_mesh_simple(
                name,
                mesh.vertices,
                mesh.faces,
                color=color_override[:3],
                opacity=color_override[3],
            )
        elif isinstance(geom, MESH_TYPES):
            frame = self._add_mesh_from_path(
                name,
                geometry_object.meshPath,
                color_override,
            )
        elif isinstance(geom, hppfcl.Convex):
            if len(geometry_object.meshPath) > 0:
                frame = self._add_mesh_from_path(
                    name,
                    geometry_object.meshPath,
                    color_override,
                )
            else:
                frame = self._add_mesh_from_convex(name, geom, color_override)
        else:
            raise RuntimeError(f"Unsupported geometry type for {name}: {type(geom)}")

        self.frames[name] = frame

    def _add_mesh_from_path(self, name, mesh_path, color):
        """Load a mesh from a file."""
        mesh = trimesh.load(mesh_path)
        if color is None:
            return self.viewer.scene.add_mesh_trimesh(name, mesh)
        else:
            return self.viewer.scene.add_mesh_simple(
                name,
                mesh.vertices,
                mesh.faces,
                color=color[:3],
                opacity=color[3],
            )

    def _add_mesh_from_convex(self, name, geom, color):
        """Load a mesh from triangles stored inside a hppfcl.Convex."""
        num_tris = geom.num_polygons
        call_triangles = geom.polygons
        call_vertices = geom.points

        vertices = call_vertices()
        vertices = vertices.astype(np.float32)
        faces = np.empty((num_tris, 3), dtype=int)
        for k in range(num_tris):
            tri = call_triangles(k)
            faces[k] = [tri[i] for i in range(3)]

        return self.viewer.scene.add_mesh_simple(
            name,
            vertices,
            faces,
            color=color[:3],
            opacity=color[3],
        )

    def display(self, q=None):
        """Display the robot at configuration q in the viewer by placing all the bodies"""
        if q is not None:
            pin.forwardKinematics(self.model, self.data, q)

        if self.collisionRootFrame.visible:
            self.updatePlacements(pin.GeometryType.COLLISION)

        if self.visualRootFrame.visible:
            self.updatePlacements(pin.GeometryType.VISUAL)

        if self.framesRootFrame.visible:
            self.updateFrames()

    def displayCollisions(self, visibility):
        self.collisionRootFrame.visible = visibility
        self.updatePlacements(pin.GeometryType.COLLISION)

    def displayVisuals(self, visibility):
        self.visualRootFrame.visible = visibility
        self.updatePlacements(pin.GeometryType.VISUAL)

    def displayFrames(self, visibility):
        self.framesRootFrame.visible = visibility
        self.updateFrames()

    def drawFrameVelocities(self, *args, **kwargs):
        raise NotImplementedError("drawFrameVelocities is not yet implemented.")

    def updatePlacements(self, geometry_type):
        if geometry_type == pin.GeometryType.VISUAL:
            geom_model = self.visual_model
            geom_data = self.visual_data
            prefix = self.viewerRootNodeName + "/visual"
        else:
            geom_model = self.collision_model
            geom_data = self.collision_data
            prefix = self.viewerRootNodeName + "/collision"

        pin.updateGeometryPlacements(self.model, self.data, geom_model, geom_data)
        for geom_id, geometry_object in enumerate(geom_model.geometryObjects):
            # Get mesh pose.
            M = geom_data.oMg[geom_id]

            # Update viewer configuration.
            frame_name = prefix + "/" + geometry_object.name
            frame = self.frames[frame_name]
            frame.position = M.translation * geometry_object.meshScale
            frame.wxyz = pin.Quaternion(M.rotation).coeffs()[
                [3, 0, 1, 2]
            ]  # Pinocchio uses xyzw

    def updateFrames(self):
        pin.updateFramePlacements(self.model, self.data)
        for frame_id, frame in enumerate(self.model.frames):
            # Get frame pose.
            M = self.data.oMf[frame_id]

            # Update viewer configuration.
            viser_frame_name = self.framesRootNodeName + "/" + frame.name
            viser_frame = self.frames[viser_frame_name]
            viser_frame.position = M.translation
            viser_frame.wxyz = pin.Quaternion(M.rotation).coeffs()[
                [3, 0, 1, 2]
            ]  # Pinocchio uses xyzw

    def captureImage(self, w=None, h=None, client_id=None, transport_format="jpeg"):
        """Capture an image from the Viser viewer and return an RGB array.

        Parameters:
            w: The width of the captured image. If None, uses the actual camera width.
            h: The height of the captured image. If None, uses the actual camera height.
            client_id: The ID of the Viser client handle.
                If None, uses the first available client.
            transport_format: The transport format to use for the captured image.
                Can be "jpeg" (default) or "png".
        """
        clients = self.viewer.get_clients()
        if len(clients) == 0:
            raise RuntimeError("Viser server has no attached clients!")

        if client_id is None:
            cli = next(iter(clients.values()))
        elif client_id not in clients:
            raise RuntimeError(
                f"Viser server does not have a client with ID '{client_id}'",
            )
        else:
            cli = clients[client_id]

        height = h or cli.camera.image_height
        width = w or cli.camera.image_width
        return cli.get_render(
            height=height,
            width=width,
            transport_format=transport_format,
        )

    def setBackgroundColor(self, preset_name: str = "gray", col_top=None, col_bot=None):
        raise NotImplementedError("setBackgroundColor is not yet implemented.")

    def setCameraTarget(self, target: np.ndarray):
        raise NotImplementedError("setCameraTarget is not yet implemented.")

    def setCameraPosition(self, position: np.ndarray):
        raise NotImplementedError("setCameraPosition is not yet implemented.")

    def setCameraZoom(self, zoom: float):
        raise NotImplementedError("setCameraZoom is not yet implemented.")

    def setCameraPose(self, pose: np.ndarray = np.eye(4)):
        raise NotImplementedError("setcameraPose is not yet implemented.")

    def disableCameraControl(self):
        raise NotImplementedError("disableCameraControl is not yet implemented.")

    def enableCameraControl(self):
        raise NotImplementedError("enableCameraControl is not yet implemented.")


class ViserVisualizer:
    """Viser visualizer for the robot."""

    def __init__(self, robot_model: RobotModel) -> None:
        """Initialize the visualizer.

        Args:
            robot_model: Robot model used for visualization.
        """
        self.robot_model = robot_model
        self.viser_visualizer = PinocchioViserVisualizer(
            robot_model.model,
            robot_model.collision_model,
            robot_model.visual_model,
        )
        self.viser_visualizer.initViewer(open=True, loadModel=True)
        self.viser_visualizer.display(pin.neutral(robot_model.model))
        self.viser_visualizer.viewer.scene.add_grid("/grid")

    def robot_state(self, robot_state: RobotState) -> None:
        """Visualize a robot state.

        Args:
            robot_state: The robot state to visualize.
        """
        check_data(self.viser_visualizer)
        self.viser_visualizer.display(robot_state.qpos)

    def robot_trajectory(self, waypoints: list[RobotState]) -> None:
        """Visualize a robot trajectory.

        Args:
            waypoints: The waypoints to visualize.
        """
        # > check_data(self.trajectory_visualizer)
        raise NotImplementedError(
            "Robot trajectory visualization is not yet implemented in ViserVisualizer.",
        )

    def frame(self, frame_name, transform):
        """Visualize a frame.

        Args:
            frame_name: The frame name to visualize.
            transform: The (4x4) transform of the frame.
        """
        raise NotImplementedError(
            "Frame visualization is not yet implemented in ViserVisualizer.",
        )

    def point(self, point_name, position):
        """Visualize a point.

        Args:
            point_name: The point name to visualize.
            position: The (3,) position of the point.
        """
        raise NotImplementedError(
            "Point visualization is not yet implemented in ViserVisualizer.",
        )
