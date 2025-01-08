# MuJoCo Simulator python interface

## Introduction

TODO

## Installation

This project is managed by [pixi](https://pixi.sh).
You can install the package in development mode using:

```bash
git clone https://github.com/JafarAbdi/mujoco_simulator_py
cd mujoco_simulator_py

pixi install -a
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

To change the logging verbosity

```bash
LOG_LEVEL=Debug pixi run python examples/...
```

### Attach models example

```bash
pixi run python examples/attach_models_demo.py mujoco_menagerie/franka_fr3/scene.xml  mujoco_menagerie/kinova_gen3/gen3.xml base_link
```
