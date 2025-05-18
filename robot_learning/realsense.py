# https://github.com/IntelRealSense/librealsense/blob/master/scripts/setup_udev_rules.sh
import cv2
import pyrealsense2
import os
import numpy as np
from dataclasses import dataclass
import loop_rate_limiters
import PIL
import timeit


CAMERAS = {
    "wrist": "145522062152",
    "scene": "251622063326",
}

import imageio
from PIL import Image
import numpy as np


def save_video_with_pillow(frames, output_path, fps=30.0):
    """
    Save a list of frames as a video using Pillow and imageio.

    Parameters:
        frames (list): List of numpy arrays or PIL Images
        output_path (str): Path to save the video file
        fps (float): Frames per second
    """

    # Write video
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Video saved to {output_path}")


import av
import numpy as np


import av
import numpy as np


def save_video_nvenc(frames, output_path, fps=30):
    """
    Save a list of numpy array frames as a video using NVIDIA's h264_nvenc encoder.
    Uses minimal settings to avoid configuration errors.

    Parameters:
        frames (list): List of numpy arrays in RGB format
        output_path (str): Path to save the video file
        fps (float): Frames per second
    """
    # Get dimensions from the first frame
    height, width = frames[0].shape[:2]

    # Create output container
    container = av.open(output_path, mode="w")

    # Add video stream with h264_nvenc codec with minimal options
    stream = container.add_stream(
        "h264_nvenc", rate=fps, options={"crf": "23", "preset": "medium"}
    )
    stream.width = width
    stream.height = height
    # stream.pix_fmt = "yuv420p"

    # Note: No additional options set to minimize chances of error

    # Encode frames
    for frame_data in frames:
        # Create AV frame from numpy array
        frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")

        # Encode and write packet
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush stream
    for packet in stream.encode():
        container.mux(packet)

    # Close the container
    container.close()
    print(f"Video saved to {output_path}")


@dataclass(frozen=True, slots=True)
class Device:
    pipeline: pyrealsense2.pipeline
    profile: pyrealsense2.pipeline_profile


class RealSenseManager:
    def __init__(self):
        self._config = pyrealsense2.config()
        self._config.enable_stream(
            pyrealsense2.stream.color,
            width=640,
            height=480,
            format=pyrealsense2.format.bgr8,
            # width=1920,
            # height=1080,
            # format=pyrealsense2.format.rgb8,
            framerate=30,
        )

        self._context = pyrealsense2.context()
        self._enabled_devices: dict[str, Device] = {}
        for device in self._context.devices:
            serial_number = device.get_info(pyrealsense2.camera_info.serial_number)
            if serial_number in CAMERAS.values():
                self._config.enable_device(serial_number)
                pipeline = pyrealsense2.pipeline()
                profile = pipeline.start(self._config)
                self._enabled_devices[serial_number] = Device(pipeline, profile)
                print(
                    f"Enabled device with serial number {serial_number} with {device.get_info(pyrealsense2.camera_info.usb_type_descriptor)}"
                )

    def poll_frames(self):
        """
        Poll for frames from the enabled Intel RealSense devices. This will return at least one frame from each device.
        """
        frames = {}
        while len(frames) < len(self._enabled_devices.items()):
            # TODO: Add a throttling message
            for serial_no, device in self._enabled_devices.items():
                streams = device.profile.get_streams()
                frameset = (
                    device.pipeline.poll_for_frames()
                )  # frameset will be a pyrealsense2.composite_frame object
                if frameset.size() == len(streams):
                    frames[serial_no] = {}
                    for stream in streams:
                        if pyrealsense2.stream.color != stream.stream_type():
                            raise RuntimeError(
                                f"Expected color stream but got {stream.stream_type()}"
                            )
                        frames[serial_no] = frameset.get_color_frame().get_data()

        return frames

    def stop(self):
        """
        Stops every device and stream
        """
        self._config.disable_all_streams()
        for serial, device in self._enabled_devices.items():
            device.pipeline.stop()
        self._enabled_devices.clear()


manager = RealSenseManager()

rate = loop_rate_limiters.RateLimiter(30)
try:
    frames = []
    import time

    start_time = time.monotonic()
    while (time.monotonic() - start_time) < 60:
        if images := manager.poll_frames():
            # if images := manager.get_frames():
            # Show images
            # print(os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"])
            cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("RealSense", np.hstack(list(images.values())))
            # cv2.imshow("RealSense", np.hstack(images))
            cv2.waitKey(1)
            frames.append(np.hstack(list(images.values())))
        rate.sleep()
    print(
        f"{timeit.timeit(lambda: save_video_with_pillow(frames, 'output.mp4'), number=1)=}"
    )
    print(
        f"{timeit.timeit(lambda: save_video_nvenc(frames, 'output_video.mp4'), number=1)=}"
    )
    print(f"Frames collected: {len(frames)}")

finally:

    # Stop streaming
    manager.stop()
