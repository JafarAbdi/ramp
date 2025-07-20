# RAMP

[![CI](https://github.com/JafarAbdi/ramp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/JafarAbdi/ramp/actions/workflows/ci.yml)

Robot-Agnostic Motion Planning Library

## Introduction

RAMP aims to be an intuitive motion planning library, designed as a simple robotics motion planning framework.

The library is built around these main components:

- **Pinocchio**: For efficient handling of rigid body kinematics.
- **OMPL**: To access a range of sampling-based planners through a simple interface.

RAMP is currently under development, working towards integrating these powerful tools into a user-friendly platform for robotics motion planning.

## Installation

This project is managed by [uv](https://docs.astral.sh/uv/getting-started/installation).
You can install the package in development mode using:

```bash
git clone https://github.com/JafarAbdi/ramp
cd ramp

uv sync
```

## Testing the package

```bash
uv run python -m pytest --capture=no
```

## Linting the package

```bash
uv run pre-commit run -a
```

## Examples

For visualization run

```bash
uv run meshcat-server --open
```

To change the logging verbosity

```bash
LOG_LEVEL=Debug uv run examples/...
```

## Acknowledgements

- Some aspects of the Pinocchio library usage were adapted from [pyroboplan](https://github.com/sea-bass/pyroboplan).
- A lot of concepts were inspired by [moveit](https://moveit.ros.org/).
