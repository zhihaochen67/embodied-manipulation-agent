# Embodied Manipulation Agent

A deterministic PyBullet project for incrementally building robotic manipulation capabilities.

**Status:** Early development — Phase 10. The Phase 8 open-loop and Phase 9 verification-only agents remain unchanged baselines.

The verification-only agent uses fresh RGB-D observations after lift and settled placement. A separate recovery-enabled agent reuses a failed verification frame to visually re-ground the cube and tray, then retries the failed task once with a fresh deterministic plan.

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
python -m embodied_manipulation.agent.closed_loop_demo
python -m embodied_manipulation.agent.recovery_demo --failure grasp
python -m embodied_manipulation.agent.recovery_demo --failure placement
```

## Tests

```bash
python -m pytest -q
```
