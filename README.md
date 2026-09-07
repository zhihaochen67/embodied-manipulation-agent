# Embodied Manipulation Agent

A deterministic PyBullet project for incrementally building robotic manipulation capabilities.

**Status:** Early development — Phase 4. The repository currently provides a deterministic PyBullet scene, inverse-kinematics reach motion, and oracle cube pick-and-place for the Franka Panda. Perception, language conditioning, verification/recovery architecture, and learned policies are not implemented.

## Setup

```bash
conda activate embodied-manip
python -m pip install -e ".[dev]"
```

## Demos

With a working graphical display:

```bash
python -m embodied_manipulation.simulation.demo
python -m embodied_manipulation.control.demo
python -m embodied_manipulation.control.pick_demo
python -m embodied_manipulation.control.pick_place_demo
```

## Tests

```bash
python -m pytest -q
```
