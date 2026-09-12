"""Curriculum manager terms for the flip task."""

from mjlab.managers.curriculum_manager import CurriculumTermCfg as CurrTerm

from wbc_ballet.utils.configclass import configclass
from . import mdp

PPO_STEPS_PER_ITERATION = 24
WALK_ONLY_ITERATIONS = 4_000
MASK_RAMP_ITERATIONS = 4_000 # was 5000
FINAL_MASK_PROBABILITY = 0.15
TARGET_SCALE_RAMP_ITERATIONS = 1_000 # was 5000
FINAL_TARGET_SCALE = 0.9

# Balance curricula start only after locomotion-only + mask-ramp training.
BALANCE_CURRICULUM_START_ITERATION = WALK_ONLY_ITERATIONS 
BALANCE_CURRICULUM_STAGE_INTERVAL_ITERATIONS = 1_000
COM_SUPPORT_PROJECTION_STAGE_WEIGHTS = (1.0 / 3.0, 2.0 / 3.0, 1.0)
PELVIS_HEIGHT_PENALTY_STAGE_WEIGHTS = (-0.5, -1.0, -2.0)
#PUSH_STAGE_HALF_RANGES = (0.15, 0.20, 0.25)

COMMANDED_LEG_CONTACT_STAGE_INTERVAL_ITERATIONS = 3_000
COMMANDED_LEG_CONTACT_STAGE_WEIGHTS = (-1.0, -2.0, -3.0, -4.0)

@configclass
class FlipCurriculumCfg:
    # Flip is a fixed 50/50 draw from the first reset; no mode curriculum.
    flip_spawn_probability: CurrTerm | None = None
#    commanded_arm_ground_contact_weight: CurrTerm | None = CurrTerm(
#        func=mdp.reward_weight_curriculum,
#       params={
#            "reward_name": "commanded_arm_ground_contact",
#            "start_steps": WALK_ONLY_ITERATIONS * PPO_STEPS_PER_ITERATION,
#            "stage_interval_steps": COMMANDED_LEG_CONTACT_STAGE_INTERVAL_ITERATIONS
#            * PPO_STEPS_PER_ITERATION,
#            "stage_weights": COMMANDED_LEG_CONTACT_STAGE_WEIGHTS,
#        },
#    )

    terrain_levels: CurrTerm | None = None
    mask_probability: CurrTerm | None = CurrTerm(
        func=mdp.mask_probability_curriculum,
        params={
            "command_name": "ballet",
            "warmup_steps": WALK_ONLY_ITERATIONS * PPO_STEPS_PER_ITERATION,
            "ramp_steps": MASK_RAMP_ITERATIONS * PPO_STEPS_PER_ITERATION,
            "initial_probability": 0.0,
            "final_probability": FINAL_MASK_PROBABILITY,
            "num_joints": 29,
        },
    )
    target_scale: CurrTerm | None = CurrTerm(
        func=mdp.target_scale_curriculum,
        params={
            "command_name": "ballet",
            "warmup_steps": WALK_ONLY_ITERATIONS * PPO_STEPS_PER_ITERATION,
            "ramp_steps": TARGET_SCALE_RAMP_ITERATIONS * PPO_STEPS_PER_ITERATION,
            "initial_scale": 0.0,
            "final_scale": FINAL_TARGET_SCALE,
        },
    )
#    pelvis_height_penalty_weight: CurrTerm | None = CurrTerm(
#        func=mdp.reward_weight_curriculum,
#        params={
#            "reward_name": "pelvis_height_penalty",
#            "start_steps": BALANCE_CURRICULUM_START_ITERATION * PPO_STEPS_PER_ITERATION,
#            "stage_interval_steps": BALANCE_CURRICULUM_STAGE_INTERVAL_ITERATIONS
#            * PPO_STEPS_PER_ITERATION,
#            "stage_weights": PELVIS_HEIGHT_PENALTY_STAGE_WEIGHTS,
#        },
#    )
#    com_support_projection_weight: CurrTerm | None = CurrTerm(
#        func=mdp.reward_weight_curriculum,
#        params={
#            "reward_name": "com_support_projection",
#            "start_steps": BALANCE_CURRICULUM_START_ITERATION * PPO_STEPS_PER_ITERATION,
#            "stage_interval_steps": BALANCE_CURRICULUM_STAGE_INTERVAL_ITERATIONS
#            * PPO_STEPS_PER_ITERATION,
#            "stage_weights": COM_SUPPORT_PROJECTION_STAGE_WEIGHTS,
#        },
#    )
#    push_velocity_range: CurrTerm | None = CurrTerm(
#        func=mdp.push_velocity_range_curriculum,
#        params={
#            "event_name": "push_robot",
#            "start_steps": BALANCE_CURRICULUM_START_ITERATION * PPO_STEPS_PER_ITERATION,
#            "stage_interval_steps": BALANCE_CURRICULUM_STAGE_INTERVAL_ITERATIONS
#            * PPO_STEPS_PER_ITERATION,
#            "stage_half_ranges": PUSH_STAGE_HALF_RANGES,
#        },
#    )
#    commanded_leg_ground_contact_weight: CurrTerm | None = CurrTerm(
#        func=mdp.reward_weight_curriculum,
#        params={
#            "reward_name": "commanded_leg_ground_contact",
#            "start_steps": WALK_ONLY_ITERATIONS * PPO_STEPS_PER_ITERATION,
#            "stage_interval_steps": COMMANDED_LEG_CONTACT_STAGE_INTERVAL_ITERATIONS
#            * PPO_STEPS_PER_ITERATION,
#            "stage_weights": COMMANDED_LEG_CONTACT_STAGE_WEIGHTS,
#        },
#    )




@configclass
class FlipPlayCurriculumCfg(FlipCurriculumCfg):
    """No active curriculum terms during play -- weights stay at their
    training end-state, evaluated by a fixed policy rather than trained."""

    terrain_levels: CurrTerm | None = None
    mask_probability: CurrTerm | None = None
    target_scale: CurrTerm | None = None
    pelvis_height_penalty_weight: CurrTerm | None = None
    com_support_projection_weight: CurrTerm | None = None
    commanded_leg_ground_contact_weight: CurrTerm | None = None
    commanded_arm_ground_contact_weight: CurrTerm | None = None
    # Play's reset_robot.flip_probability=None (follow the UDP checkbox) must
    # not get overwritten by the ramped value -- see flip_events_cfg.py.
    flip_spawn_probability: CurrTerm | None = None
