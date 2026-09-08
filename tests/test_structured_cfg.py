import os

os.environ.setdefault("MUJOCO_GL", "glfw")

from wbc_ballet import mdp as ballet_mdp
from wbc_ballet.tasks.ballet import make_ballet_env_cfg
from wbc_ballet.tasks.ballet.ballet_commands_cfg import BalletCommandsCfg
from wbc_ballet.tasks.ballet.ballet_curriculum_cfg import (
    FINAL_MASK_PROBABILITY,
    FINAL_TARGET_SCALE,
    MASK_RAMP_ITERATIONS,
    PPO_STEPS_PER_ITERATION,
    TARGET_SCALE_RAMP_ITERATIONS,
    WALK_ONLY_ITERATIONS,
)
from wbc_ballet.tasks.ballet.ballet_env_cfg import BalletEnvCfg
from wbc_ballet.tasks.ballet.ballet_observations_cfg import BalletObservationsCfg
from wbc_ballet.tasks.ballet.ballet_rl_cfg import ballet_ppo_runner_cfg

EXPECTED_ACTOR_TERMS = [
    "base_lin_vel",
    "base_ang_vel",
    "imu_lin_acc",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
    "axis_actual_normalized",
    "axis_target_normalized",
    "axis_mask",
]


def test_manager_terms_are_converted_from_configclasses() -> None:
    declarative = BalletEnvCfg()
    assert isinstance(declarative.observations, BalletObservationsCfg)

    native = declarative.to_mjlab_cfg()
    assert list(native.observations["actor"].terms) == EXPECTED_ACTOR_TERMS
    assert "whole_body_com_xy" not in native.observations["actor"].terms
    assert "whole_body_com_xy" in native.observations["critic"].terms
    assert "support_center_xy" not in native.observations["actor"].terms
    assert "support_center_xy" in native.observations["critic"].terms
    assert list(native.actions) == ["joint_pos"]
    assert list(native.commands) == ["ballet"]
    assert "masked_pose_tracking" in native.rewards
    assert "pelvis_height_tracking" in native.rewards
    assert "leg_lateral_alignment" in native.rewards
    assert "foot_heading_alignment" in native.rewards
    assert "com_support_projection" in native.rewards
    assert "commanded_leg_ground_contact" in native.rewards
    assert native.rewards["commanded_leg_ground_contact"].weight == -1.0
    assert native.rewards["track_angular_velocity"].weight == 3.0
    assert (
        native.rewards["commanded_leg_ground_contact"].params["velocity_epsilon"]
        == ballet_mdp.VELOCITY_EPSILON
    )
    assert (
        native.rewards["com_support_projection"].params["velocity_epsilon"]
        == ballet_mdp.VELOCITY_EPSILON
    )
    assert "non_finite_state_penalty" in native.rewards
    assert "mask_probability" in native.curriculum
    assert "target_scale" in native.curriculum
    assert native.terminations["fell_over"].func is ballet_mdp.pelvis_height_below
    assert native.terminations["fell_over"].params["minimum_height"] == 0.2
    assert native.terminations["non_finite_state"].func is ballet_mdp.non_finite_state_or_action
    assert "nan_physics_state" in native.metrics
    assert "inf_physics_state" in native.metrics
    assert "nonfinite_policy_action" in native.metrics
    assert native.observations["actor"].nan_policy == "warn"
    assert native.observations["actor"].nan_check_per_term is False
    assert native.observations["critic"].nan_policy == "warn"
    assert native.observations["critic"].nan_check_per_term is False
    assert native.sim.nan_guard.enabled is False
    assert native.sim.nan_guard.output_dir == "logs/nan_dumps"


def test_config_instances_do_not_share_mutable_terms() -> None:
    first = BalletEnvCfg()
    second = BalletEnvCfg()
    first.rewards.masked_pose_tracking.weight = 123.0
    assert second.rewards.masked_pose_tracking.weight == 2.5


def test_train_and_play_keep_observation_abi() -> None:
    train_cfg = make_ballet_env_cfg()
    play_cfg = make_ballet_env_cfg(play=True)
    assert list(train_cfg.observations["actor"].terms) == list(play_cfg.observations["actor"].terms)
    assert train_cfg.scene.num_envs == 4096
    assert play_cfg.scene.num_envs == 1
    assert "joint_observation_table" not in train_cfg.events
    assert "joint_observation_table" in play_cfg.events
    assert play_cfg.events["joint_observation_table"].params["env_index"] == 0
    assert "mask_probability" in train_cfg.curriculum
    assert "target_scale" in train_cfg.curriculum
    assert "mask_probability" not in play_cfg.curriculum
    assert "target_scale" not in play_cfg.curriculum


def test_policy_target_observation_is_hard_masked() -> None:
    cfg = make_ballet_env_cfg()
    actor_term = cfg.observations["actor"].terms["axis_target_normalized"]
    critic_term = cfg.observations["critic"].terms["axis_target_normalized"]
    assert actor_term.func is ballet_mdp.masked_ballet_targets
    assert critic_term.func is ballet_mdp.masked_ballet_targets


def test_ballet_curricula_are_aligned_with_ppo_iterations() -> None:
    curriculum = BalletEnvCfg().curriculum
    mask_term = curriculum.mask_probability
    scale_term = curriculum.target_scale
    assert mask_term is not None
    assert scale_term is not None
    assert mask_term.params["warmup_steps"] == WALK_ONLY_ITERATIONS * PPO_STEPS_PER_ITERATION
    assert mask_term.params["ramp_steps"] == MASK_RAMP_ITERATIONS * PPO_STEPS_PER_ITERATION
    assert mask_term.params["final_probability"] == FINAL_MASK_PROBABILITY
    assert scale_term.params["warmup_steps"] == WALK_ONLY_ITERATIONS * PPO_STEPS_PER_ITERATION
    assert scale_term.params["ramp_steps"] == (
        TARGET_SCALE_RAMP_ITERATIONS * PPO_STEPS_PER_ITERATION
    )
    assert scale_term.params["final_scale"] == FINAL_TARGET_SCALE
    assert ballet_ppo_runner_cfg().num_steps_per_env == PPO_STEPS_PER_ITERATION


def test_training_command_starts_with_zero_masks_and_safe_targets() -> None:
    command = BalletCommandsCfg().ballet
    assert command is not None
    assert command.resampling_time_range == (6.0, 10.0)
    assert command.mask_probability == 0.0
    assert command.target_scale == 0.0
    assert command.target_limit == 0.8
    assert command.velocity_ranges == ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))


def test_training_episode_period_is_24_seconds() -> None:
    assert BalletEnvCfg().episode_length_s == 24.0
