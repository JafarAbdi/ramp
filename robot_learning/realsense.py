"""Visualize the RealSense camera stream."""

# https://github.com/IntelRealSense/librealsense/blob/master/scripts/setup_udev_rules.sh
import logging
from dataclasses import dataclass

import cv2
import loop_rate_limiters
import numpy as np
import pyrealsense2

LOGGER = logging.getLogger(__name__)

CAMERAS = {
    "wrist": "145522062152",
    "scene": "251622063326",
}


@dataclass(frozen=True, slots=True)
class Device:
    """A dataclass to hold the pipeline and profile of a Realsense device."""

    pipeline: pyrealsense2.pipeline
    profile: pyrealsense2.pipeline_profile


class RealSenseManager:
    """Manager for Intel Realsense cameras."""

    def __init__(self):
        """Initialize the RealSenseManager and enable the devices."""
        self._config = pyrealsense2.config()
        self._config.enable_stream(
            pyrealsense2.stream.color,
            width=640,
            height=480,
            format=pyrealsense2.format.bgr8,
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
                LOGGER.info(
                    f"Enabled device with serial number {serial_number} with {device.get_info(pyrealsense2.camera_info.usb_type_descriptor)}",
                )

    def poll_frames(self):
        """Poll for frames from the enabled Intel RealSense devices. This will return at least one frame from each device."""
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
                            msg = (
                                f"Expected color stream but got {stream.stream_type()}"
                            )
                            raise RuntimeError(msg)
                        frames[serial_no] = frameset.get_color_frame().get_data()

        return frames

    def stop(self):
        """Stops every device and stream."""
        self._config.disable_all_streams()
        for _serial, device in self._enabled_devices.items():
            device.pipeline.stop()
        self._enabled_devices.clear()


manager = RealSenseManager()

rate = loop_rate_limiters.RateLimiter(30)
try:
    while True:
        if images := manager.poll_frames():
            # Show images
            cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("RealSense", np.hstack(list(images.values())))
            cv2.waitKey(1)
        rate.sleep()

finally:
    # Stop streaming
    manager.stop()
