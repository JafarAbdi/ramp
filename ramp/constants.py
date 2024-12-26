"""This module contains the constants used in the package."""

# Loading from robot_descriptions
ROBOT_DESCRIPTION_PREFIX = "robot-descriptions::"
MUJOCO_DESCRIPTION_VARIANT = "_mj_"

# Differential-IK Hyperparameters
MAX_ITERATIONS = 500
MAX_ERROR = 1e-3

# Pinocchi joint patterns
PINOCCHIO_PLANAR_JOINT = "JointModelPlanar"
PINOCCHIO_REVOLUTE_JOINT = r"JointModelR([XYZ])$"
PINOCCHIO_PRISMATIC_JOINT = r"JointModelP([XYZ])$"
PINOCCHIO_UNBOUNDED_JOINT = r"JointModelRUB([XYZ])$"
PINOCCHIO_SPHERICAL_JOINT = "JointModelSphericalZYX"
PINOCCHIO_TRANSLATION_JOINT = "JointModelTranslation"
PINOCCHIO_FREEFLYER_JOINT = "JointModelFreeFlyer"
PINOCCHIO_REVOLUTE_UNALIGNED_JOINT = "JointModelRevoluteUnaligned"
# JointModelRevoluteUnboundedUnaligned
# JointModelPrismaticUnaligned
