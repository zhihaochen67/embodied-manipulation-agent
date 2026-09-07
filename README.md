# Embodied Manipulation Agent

A simulation-based vision-language-conditioned robotic manipulation agent with task planning, closed-loop execution, verification, and failure recovery.

**Status:** Early development — Phase 3. The repository currently provides a deterministic PyBullet scene, inverse-kinematics reach motion, and an oracle top-down cube pick for the Franka Panda. Placement, perception, language conditioning, and recovery are not implemented.

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
```

## Tests

```bash
python -m pytest -q
```
