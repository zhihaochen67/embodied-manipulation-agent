# Embodied Manipulation Agent

A simulation-based vision-language-conditioned robotic manipulation agent with task planning, closed-loop execution, verification, and failure recovery.

**Status:** Early development — Phase 2. The repository currently provides a deterministic PyBullet scene and inverse-kinematics reach motion for the Franka Panda. Grasping and object manipulation are not implemented.

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
```

## Tests

```bash
python -m pytest -q
```
