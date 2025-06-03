"""Keyboard listener for controlling the robot."""

import logging
from dataclasses import dataclass, field
from threading import Lock, Thread

from sshkeyboard import listen_keyboard, stop_listening

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class KeyboardEvents:
    """A class to hold keyboard events."""

    record_request: bool = False
    stop_request: bool = False
    discard_request: bool = False
    exit_request: bool = False
    reset_robot_request: bool = False
    # Used to communicate with LeRobot
    dict_events: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize the events dictionary."""
        self.dict_events = {
            "exit_early": False,
        }


class KeyboardListener:
    """A class to listen to keyboard events."""

    def __init__(self):
        """Initialize the keyboard listener."""
        self.events = KeyboardEvents()
        self._mutex = Lock()
        self._keyboard_listener_thread = Thread(
            target=listen_keyboard,
            args=(self._on_key_press,),
            daemon=True,
        )
        self._keyboard_listener_thread.start()
        LOGGER.info("Keyboard Controls:")
        LOGGER.info("  p: Reset robot.")
        LOGGER.info("  r: Record episode.")
        LOGGER.info("  s: Stop episode.")
        LOGGER.info("  d: Discard episode.")
        LOGGER.info("  q: Quit.")

    def _on_key_press(self, key):
        """Handle key press from the keyboard."""
        with self._mutex:
            if key == "r":
                self.events.record_request = True
            elif key == "s":
                self.events.stop_request = True
                self.events.dict_events["exit_early"] = True
            elif key == "d":
                self.events.discard_request = True
                self.events.dict_events["exit_early"] = True
            elif key == "q":
                self.events.exit_request = True
            elif key == "p":
                self.events.reset_robot_request = True

    def start_recording(self):
        """Start recording an episode."""
        with self._mutex:
            record_request, self.events.record_request = (
                self.events.record_request,
                False,
            )
        return record_request

    def stop_recording(self):
        """Stop recording an episode."""
        with self._mutex:
            stop_request, self.events.stop_request = (
                self.events.stop_request,
                False,
            )
        return stop_request

    def discard_recording(self):
        """Discard the current recording."""
        with self._mutex:
            discard_request, self.events.discard_request = (
                self.events.discard_request,
                False,
            )
        return discard_request

    def exit(self):
        """Exit the program."""
        with self._mutex:
            exit_request, self.events.exit_request = (
                self.events.exit_request,
                False,
            )
        return exit_request

    def reset_robot(self):
        """Reset the robot."""
        with self._mutex:
            reset_request, self.events.reset_robot_request = (
                self.events.reset_robot_request,
                False,
            )
        return reset_request

    def stop(self):
        """Stop the keyboard listener."""
        stop_listening()
        self._keyboard_listener_thread.join()
