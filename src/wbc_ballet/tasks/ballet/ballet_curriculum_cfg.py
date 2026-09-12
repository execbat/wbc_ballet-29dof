"""Curriculum manager terms for G1 ballet."""

from mjlab.managers.curriculum_manager import CurriculumTermCfg as CurrTerm
from mjlab.tasks.velocity import mdp

from wbc_ballet.tasks.ballet import mdp as ballet_mdp
from wbc_ballet.utils.configclass import configclass

# These values are intentionally explicit and shared with the PPO runner.
# MJLab exposes environment steps to curriculum terms, so PPO iterations are
# converted using the rollout length (num_steps_per_env).
PPO_STEPS_PER_ITERATION = 24
WALK_ONLY_ITERATIONS = 1_000
MASK_RAMP_ITERATIONS = 1_000 # was 5000
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
class BalletCurriculumCfg:
    terrain_levels: CurrTerm | None = None
    mask_probability: CurrTerm | None = CurrTerm(
        func=ballet_mdp.mask_probability_curriculum,
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
        func=ballet_mdp.target_scale_curriculum,
        params={
            "command_name": "ballet",
            "warmup_steps": WALK_ONLY_ITERATIONS * PPO_STEPS_PER_ITERATION,
            "ramp_steps": TARGET_SCALE_RAMP_ITERATIONS * PPO_STEPS_PER_ITERATION,
            "initial_scale": 0.0,
            "final_scale": FINAL_TARGET_SCALE,
        },
    )
    pelvis_height_penalty_weight: CurrTerm | None = CurrTerm(
        func=ballet_mdp.reward_weight_curriculum,
        params={
            "reward_name": "pelvis_height_penalty",
            "start_steps": BALANCE_CURRICULUM_START_ITERATION * PPO_STEPS_PER_ITERATION,
            "stage_interval_steps": BALANCE_CURRICULUM_STAGE_INTERVAL_ITERATIONS
            * PPO_STEPS_PER_ITERATION,
            "stage_weights": PELVIS_HEIGHT_PENALTY_STAGE_WEIGHTS,
        },
    )
    com_support_projection_weight: CurrTerm | None = CurrTerm(
        func=ballet_mdp.reward_weight_curriculum,
        params={
            "reward_name": "com_support_projection",
            "start_steps": BALANCE_CURRICULUM_START_ITERATION * PPO_STEPS_PER_ITERATION,
            "stage_interval_steps": BALANCE_CURRICULUM_STAGE_INTERVAL_ITERATIONS
            * PPO_STEPS_PER_ITERATION,
            "stage_weights": COM_SUPPORT_PROJECTION_STAGE_WEIGHTS,
        },
    )
#    push_velocity_range: CurrTerm | None = CurrTerm(
#        func=ballet_mdp.push_velocity_range_curriculum,
#        params={
#            "event_name": "push_robot",
#            "start_steps": BALANCE_CURRICULUM_START_ITERATION * PPO_STEPS_PER_ITERATION,
#            "stage_interval_steps": BALANCE_CURRICULUM_STAGE_INTERVAL_ITERATIONS
#            * PPO_STEPS_PER_ITERATION,
#            "stage_half_ranges": PUSH_STAGE_HALF_RANGES,
#        },
#    )
    commanded_leg_ground_contact_weight: CurrTerm | None = CurrTerm(
        func=ballet_mdp.reward_weight_curriculum,
        params={
            "reward_name": "commanded_leg_ground_contact",
            "start_steps": 0,
            "stage_interval_steps": COMMANDED_LEG_CONTACT_STAGE_INTERVAL_ITERATIONS
            * PPO_STEPS_PER_ITERATION,
            "stage_weights": COMMANDED_LEG_CONTACT_STAGE_WEIGHTS,
        },
    )

@configclass
class BalletRoughCurriculumCfg(BalletCurriculumCfg):
    terrain_levels: CurrTerm | None = CurrTerm(
        func=mdp.terrain_levels_vel,
        params={"command_name": "ballet"},
    )


@configclass
class BalletPlayCurriculumCfg(BalletCurriculumCfg):
    terrain_levels: CurrTerm | None = None
    mask_probability: CurrTerm | None = None
    target_scale: CurrTerm | None = None
    pelvis_height_penalty_weight: CurrTerm | None = None
    com_support_projection_weight: CurrTerm | None = None
#    push_velocity_range: CurrTerm | None = None
