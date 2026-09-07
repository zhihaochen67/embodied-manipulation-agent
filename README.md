# Embodied Manipulation Agent

A simulation-based vision-language-conditioned robotic manipulation agent with task planning, closed-loop execution, verification, and failure recovery.

**Status:** Early development — Phase 1. The repository currently provides a minimal deterministic PyBullet scene with a Franka Panda, table, cube, and fixed camera.

## Setup

```bash
conda activate embodied-manip
python -m pip install -e ".[dev]"
```

## Demo

With a working graphical display:

```bash
python -m embodied_manipulation.simulation.demo
```

## Tests

```bash
python -m pytest -q
```
