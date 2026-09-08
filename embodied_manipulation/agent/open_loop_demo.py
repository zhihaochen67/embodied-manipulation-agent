"""GUI demonstration of the Phase 8 vision open-loop agent."""

from __future__ import annotations

from math import dist
from time import sleep

from embodied_manipulation.benchmark import generate_scenario
from embodied_manipulation.perception import LocalizationResult
from embodied_manipulation.planning import ManipulationPlan
from embodied_manipulation.simulation import World

from .open_loop import VisionOpenLoopAgent

GUI_INITIAL_PAUSE = 1.0
GUI_STAGE_PAUSE = 0.35
GUI_STEP_DELAY = 1.0 / 240.0
DEMO_SEED = 0
DEMO_DISTRACTOR_COUNT = 2


def main() -> None:
    """Run one predeclared scenario; no re-observation or retry is performed."""
    scenario = generate_scenario(
        DEMO_SEED,
        distractor_count=DEMO_DISTRACTOR_COUNT,
    )
    print(f"instruction: {scenario.task.instruction}")
    print(f"scenario seed: {DEMO_SEED}")
    print(f"distractor count: {len(scenario.distractors)}")

    try:
        with World(gui=True) as world:
            world.reset_from_scenario(scenario)
            sleep(GUI_INITIAL_PAUSE)
            result = VisionOpenLoopAgent().run(
                scenario.task.instruction,
                world,
                step_delay=GUI_STEP_DELAY,
                stage_pause=GUI_STAGE_PAUSE,
                stage_callback=_print_stage,
            )

            if result.source_detection is not None:
                print(
                    "cube vision error (post-run diagnostic only): "
                    f"{dist(result.source_detection.world_position, scenario.source.initial_position):.6f} m"
                )
            if result.target_detection is not None:
                print(
                    "tray vision error (post-run diagnostic only): "
                    f"{dist(result.target_detection.world_position, scenario.target.initial_position):.6f} m"
                )
            print(
                "execution source: vision; "
                "final evaluation source: simulator ground truth"
            )
            print(f"execution success: {result.execution_success}")
            print(f"final task success: {result.success}")
            print(f"failure stage: {result.failure_stage}")
            print(f"failure reason: {result.failure_reason}")
            print(f"total simulation steps: {result.total_steps}")

            while world.is_connected:
                world.step()
                sleep(world.time_step)
    except KeyboardInterrupt:
        pass


def _print_stage(stage: str, value: object) -> None:
    if stage == "parsed":
        task = value
        print(
            "parsed source/target: "
            f"{task.source.color} {task.source.object_type} -> "
            f"{task.target.color} {task.target.object_type}"
        )
    elif stage == "observation_captured":
        print("captured initial RGB-D observation (1/1)")
    elif stage == "source_grounded":
        detection = value
        assert isinstance(detection, LocalizationResult)
        print(f"vision cube estimate: {detection.world_position}")
    elif stage == "target_grounded":
        detection = value
        assert isinstance(detection, LocalizationResult)
        print(f"vision tray estimate: {detection.world_position}")
    elif stage == "planned":
        plan = value
        assert isinstance(plan, ManipulationPlan)
        print(f"planned grasp target: {plan.grasp_target}")
        print(f"planned release target: {plan.release_target}")


if __name__ == "__main__":
    main()
