"""Curriculum manager terms for the flip task."""

from mjlab.managers.curriculum_manager import CurriculumTermCfg as CurrTerm

from wbc_ballet.utils.configclass import configclass
from . import mdp
from mjlab.envs import mdp as env_mdp

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

# Standing-form regularizers start weak immediately, then become stricter.
# This prevents the policy from first learning a crooked standing local optimum
# and only later being asked to unlearn it.
POSTURE_STAGE_INTERVAL_ITERATIONS = 1_000
FEET_STAND_POSE_STAGE_WEIGHTS = (-0.25, -0.50, -0.75, -1.00)
FEET_FLATNESS_STAGE_WEIGHTS = (-0.10, -0.20, -0.35, -0.50)
PELVIS_ORIENTATION_STAGE_WEIGHTS = (-0.50, -1.00, -1.50, -2.00)
WAIST_ZERO_STAGE_WEIGHTS = (-1.00, -2.00, -3.00, -4.00)
HANDSTAND_LEG_POSE_STAGE_WEIGHTS = (-0.25, -0.50, -1.00, -1.50)

FORBIDDEN_CONTACT_TERMINATION_ITERATIONS = 10000

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

    feet_stand_pose_hold_weight: CurrTerm | None = CurrTerm(
        func=mdp.reward_weight_curriculum,
        params={
            "reward_name": "feet_stand_pose_hold",
            "start_steps": 0,
            "stage_interval_steps": POSTURE_STAGE_INTERVAL_ITERATIONS * PPO_STEPS_PER_ITERATION,
            "stage_weights": FEET_STAND_POSE_STAGE_WEIGHTS,
        },
    )
    feet_flatness_penalty_weight: CurrTerm | None = CurrTerm(
        func=mdp.reward_weight_curriculum,
        params={
            "reward_name": "feet_flatness_penalty",
            "start_steps": 0,
            "stage_interval_steps": POSTURE_STAGE_INTERVAL_ITERATIONS * PPO_STEPS_PER_ITERATION,
            "stage_weights": FEET_FLATNESS_STAGE_WEIGHTS,
        },
    )
    pelvis_orientation_penalty_weight: CurrTerm | None = CurrTerm(
        func=mdp.reward_weight_curriculum,
        params={
            "reward_name": "pelvis_orientation_penalty",
            "start_steps": 0,
            "stage_interval_steps": POSTURE_STAGE_INTERVAL_ITERATIONS * PPO_STEPS_PER_ITERATION,
            "stage_weights": PELVIS_ORIENTATION_STAGE_WEIGHTS,
        },
    )
    waist_zero_pose_penalty_weight: CurrTerm | None = CurrTerm(
        func=mdp.reward_weight_curriculum,
        params={
            "reward_name": "waist_zero_pose_penalty",
            "start_steps": 0,
            "stage_interval_steps": POSTURE_STAGE_INTERVAL_ITERATIONS * PPO_STEPS_PER_ITERATION,
            "stage_weights": WAIST_ZERO_STAGE_WEIGHTS,
        },
    )
    handstand_leg_pose_penalty_weight: CurrTerm | None = CurrTerm(
        func=mdp.reward_weight_curriculum,
        params={
            "reward_name": "handstand_leg_pose_penalty",
            "start_steps": 0,
            "stage_interval_steps": POSTURE_STAGE_INTERVAL_ITERATIONS * PPO_STEPS_PER_ITERATION,
            "stage_weights": HANDSTAND_LEG_POSE_STAGE_WEIGHTS,
        },
    )
    
    # SWITCH OFF FORBIDDEN TERMINATION
    forbidden_ground_contact_termination: CurrTerm | None = CurrTerm(
        func=env_mdp.termination_curriculum,
        params={
            "termination_name": "forbidden_ground_contact",
            "stages": [
                {
                    "step": 0,
                    "params": {
                        "enabled": True,
                    },
                },
                {
                    "step": (
                        FORBIDDEN_CONTACT_TERMINATION_ITERATIONS
                        * PPO_STEPS_PER_ITERATION
                    ),
                    "params": {
                        "enabled": False,
                    },
                },
            ],
        },
    )    
    forbidden_contact_termination_penalty_weight: CurrTerm | None = CurrTerm(
        func=mdp.reward_weight_curriculum,
        params={
            "reward_name": "forbidden_contact_termination_penalty",
            "start_steps": 0,
            "stage_interval_steps": (
                FORBIDDEN_CONTACT_TERMINATION_ITERATIONS
                * PPO_STEPS_PER_ITERATION
            ),
            "stage_weights": (
                -100.0,
                0.0,
            ),
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
    feet_stand_pose_hold_weight: CurrTerm | None = None
    feet_flatness_penalty_weight: CurrTerm | None = None
    pelvis_orientation_penalty_weight: CurrTerm | None = None
    waist_zero_pose_penalty_weight: CurrTerm | None = None
    handstand_leg_pose_penalty_weight: CurrTerm | None = None
    pelvis_height_penalty_weight: CurrTerm | None = None
    com_support_projection_weight: CurrTerm | None = None
    commanded_leg_ground_contact_weight: CurrTerm | None = None
    commanded_arm_ground_contact_weight: CurrTerm | None = None
    # Play's reset_robot.flip_probability=None (follow the UDP checkbox) must
    # not get overwritten by the ramped value -- see flip_events_cfg.py.
    flip_spawn_probability: CurrTerm | None = None
