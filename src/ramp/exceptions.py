"""Exceptions for ramp package."""


class MissingBaseLinkError(Exception):
    """Missing base link error."""


class MissingJointError(Exception):
    """Missing joint error."""


class RobotDescriptionNotFoundError(Exception):
    """Robot description error."""

    def __init__(self, robot_description_name):
        """Init.

        Args:
            robot_description_name: Name of the robot description
        """
        message = (
            f"Failed to import robot description '{robot_description_name}' from robot_descriptions. "
            "See https://github.com/robot-descriptions/robot_descriptions.py for the available robot descriptions."
        )
        super().__init__(message)


class MissingAccelerationLimitError(Exception):
    """Acceleration limits error."""

    def __init__(self, joint_names, defined_joint_names):
        """Init.

        Args:
            joint_names: List of joint names
            defined_joint_names: List of defined joint names
        """
        message = f"Acceleration limits only defined for {defined_joint_names} - Missing {set(joint_names) - set(defined_joint_names)}"
        super().__init__(message)
