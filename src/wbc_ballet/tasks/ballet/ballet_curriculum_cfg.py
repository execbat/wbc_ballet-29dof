"""Curriculum manager terms for G1 ballet."""

from mjlab.managers.curriculum_manager import CurriculumTermCfg as CurrTerm
from mjlab.tasks.velocity import mdp

from wbc_ballet import mdp as ballet_mdp
from wbc_ballet.utils.configclass import configclass

# These values are intentionally explicit and shared with the PPO runner.
# MJLab exposes environment steps to curriculum terms, so PPO iterations are
# converted using the rollout length (num_steps_per_env).
PPO_STEPS_PER_ITERATION = 24
WALK_ONLY_ITERATIONS = 300
MASK_RAMP_ITERATIONS = 5_000
FINAL_MASK_PROBABILITY = 0.15
TARGET_SCALE_RAMP_ITERATIONS = 5_000
FINAL_TARGET_SCALE = 0.8


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
