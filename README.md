# Embodied Manipulation Agent

A deterministic PyBullet project for incrementally building robotic manipulation capabilities.

**Status:** Early development — Phase 7. The repository currently provides deterministic PyBullet scenes, structured task metadata, seeded scenario generation, oracle cube pick-and-place for the Franka Panda, deterministic parsing for a bounded family of tabletop instructions, and color-based RGB-D localization of cubes and trays with image-to-world back-projection. General language understanding, learned or open-vocabulary vision, end-to-end agent execution, benchmark execution, and verification/recovery architecture are not implemented.

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
