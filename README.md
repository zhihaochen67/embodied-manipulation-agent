# Embodied Manipulation Agent

A deterministic PyBullet project for incrementally building robotic manipulation capabilities.

**Status:** Early development — Phase 6. The repository currently provides deterministic PyBullet scenes, structured task metadata, seeded scenario generation, oracle cube pick-and-place for the Franka Panda, and deterministic parsing for a bounded family of tabletop instructions. General language understanding, perception, benchmark execution, verification/recovery architecture, and learned policies are not implemented.

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
