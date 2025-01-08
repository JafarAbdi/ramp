# RAMP

[![CI](https://github.com/JafarAbdi/ramp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/JafarAbdi/ramp/actions/workflows/ci.yml)

Robot-Agnostic Motion Planning Library

## Introduction

RAMP aims to be an intuitive motion planning library, designed as a simple robotics motion planning framework.

The library is built around these main components:

- **Pinocchio**: For efficient handling of rigid body kinematics.
- **OMPL**: To access a range of sampling-based planners through a simple interface.
- **MuJoCo**: To set up and run accurate physics simulations.

RAMP is currently under development, working towards integrating these powerful tools into a user-friendly platform for robotics motion planning.

## Installation

This project is managed by [pixi](https://pixi.sh).
You can install the package in development mode using:

```bash
git clone https://github.com/JafarAbdi/ramp
cd ramp

pixi install -a
pixi run install-mujoco
```

## Testing the package

```bash
pixi run test
```

## Linting the package

```bash
pixi run lint
```

## Examples

For visualization run

```bash
pixi run meshcat
```

To change the logging verbosity

```bash
LOG_LEVEL=Debug pixi run python examples/...
```

To change the logging verbosity for trac-ik solver

```
SPDLOG_LEVEL=debug pixi run python XXX
```

### Attach models example

```bash
pixi run python ./examples/attach_models_demo.py ./external/mujoco_menagerie/unitree_a1/a1.xml ./external/mujoco_menagerie/trs_so_arm100/so_arm100.xml Base
```

## Acknowledgements

- Some aspects of the Pinocchio library usage were adapted from [pyroboplan](https://github.com/sea-bass/pyroboplan).
- A lot of concepts were inspired by [moveit](https://moveit.ros.org/).
