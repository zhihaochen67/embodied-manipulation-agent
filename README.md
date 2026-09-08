# Embodied Manipulation Agent

A deterministic PyBullet project for incrementally building robotic manipulation capabilities.

**Status:** Early development — Phase 8. The repository now includes a vision-conditioned open-loop manipulation agent that parses a bounded tabletop instruction, captures one RGB-D observation, grounds a cube and tray, plans fixed Cartesian manipulation targets, and physically executes pick-and-place with the Franka Panda. Execution does not re-observe, verify, retry, recover, or use a learned policy.

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
python -m embodied_manipulation.agent.open_loop_demo
```

## Tests

```bash
python -m pytest -q
```
