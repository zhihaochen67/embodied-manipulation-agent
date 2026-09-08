"""GUI demonstration of Phase 9 stage-level visual verification."""

from __future__ import annotations

from math import dist
from time import sleep

from embodied_manipulation.benchmark import generate_scenario
from embodied_manipulation.simulation import World

from .closed_loop import VisionClosedLoopAgent
from .verification import GraspVerificationResult, PlacementVerificationResult

GUI_INITIAL_PAUSE = 1.0
GUI_STAGE_PAUSE = 0.35
GUI_STEP_DELAY = 1.0 / 240.0
DEMO_SEED = 0
DEMO_DISTRACTOR_COUNT = 2


def main() -> None:
    """Run one deterministic three-observation episode without recovery."""
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
            result = VisionClosedLoopAgent().run(
                scenario.task.instruction,
                world,
                step_delay=GUI_STEP_DELAY,
                stage_pause=GUI_STAGE_PAUSE,
                stage_callback=_print_stage,
            )

            if result.source_detection is not None:
                print(
                    "initial cube vision error (diagnostic only): "
                    f"{dist(result.source_detection.world_position, scenario.source.initial_position):.6f} m"
                )
            if result.target_detection is not None:
                print(
                    "initial tray vision error (diagnostic only): "
                    f"{dist(result.target_detection.world_position, scenario.target.initial_position):.6f} m"
                )
            print(f"final objective success: {bool(result.final_evaluation and result.final_evaluation.success)}")
            print(f"closed-loop success: {result.success}")
            print(f"failure stage: {result.failure_stage}")
            print(f"failure reason: {result.failure_reason}")
            print(f"observation count: {result.observation_count}")
            print(f"total simulation steps: {result.total_steps}")

            while world.is_connected:
                world.step()
                sleep(world.time_step)
    except KeyboardInterrupt:
        pass


def _print_stage(stage: str, value: object) -> None:
    if stage == "initial_observation_captured":
        print("captured initial RGB-D observation (1/3)")
    elif stage == "source_grounded":
        print(f"initial source estimate: {value.world_position}")
    elif stage == "target_grounded":
        print(f"initial target estimate: {value.world_position}")
    elif stage == "post_grasp_observation_captured":
        print("captured post-grasp RGB-D observation (2/3)")
    elif stage == "grasp_verified":
        result = value
        assert isinstance(result, GraspVerificationResult)
        print(f"grasp verification cube position: {result.estimated_cube_position}")
        print(f"cube bottom height above table: {result.cube_height_above_table}")
        print(f"cube-to-EE distance: {result.cube_to_ee_distance}")
        print(f"grasp verified: {result.verified}")
        print("GRASP VERIFIED" if result.verified else "GRASP REJECTED")
    elif stage == "post_placement_observation_captured":
        print("captured post-placement RGB-D observation (3/3)")
    elif stage == "placement_verified":
        result = value
        assert isinstance(result, PlacementVerificationResult)
        print(f"placement cube position: {result.estimated_cube_position}")
        print(f"placement tray position: {result.estimated_tray_position}")
        print(f"inside tray: {result.cube_inside_tray}")
        print(f"placement verified: {result.verified}")
        print(
            "PLACEMENT VERIFIED"
            if result.verified
            else "PLACEMENT REJECTED"
        )
    elif stage == "evaluated":
        print("SUCCESS" if value.success else "OBJECTIVE FAILURE")


if __name__ == "__main__":
    main()
