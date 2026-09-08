# Embodied Manipulation Agent

A deterministic PyBullet project for incrementally building robotic manipulation capabilities.

**Status:** Early development — Phase 9. The Phase 8 vision-conditioned open-loop agent remains an observe-once baseline.

The separate closed-loop agent uses fresh RGB-D observations after lift and after settled placement to verify grasp and tray containment. Verification can only continue or stop the immutable initial plan; it does not retry, recover, or replan.

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
```

## Tests

```bash
python -m pytest -q
```
