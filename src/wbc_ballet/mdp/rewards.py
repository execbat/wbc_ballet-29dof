from __future__ import annotations

import torch
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from .observations import (
    _group_has_active_mask,
    ballet_mask,
    ballet_targets,
    joint_pos_normalized,
    support_center_xy_b,
    whole_body_com_xy_b,
)

_ROBOT_CFG = SceneEntityCfg("robot")
VELOCITY_EPSILON = 0.01


def _selected_command_axes(values: torch.Tensor, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Select command axes using the joint ids resolved by ``SceneEntityCfg``."""
    return values[:, asset_cfg.joint_ids]


def track_ballet_linear_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("ballet")[:, :3]
    actual = robot.data.root_link_lin_vel_b
    error = (command[:, :2] - actual[:, :2]).square().sum(dim=1) + actual[:, 2].square()
    return torch.exp(-error / (std * std))


def track_ballet_angular_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("ballet")[:, :3]
    actual = robot.data.root_link_ang_vel_b
    error = (command[:, 2] - actual[:, 2]).square() + actual[:, :2].square().sum(dim=1)
    return torch.exp(-error / (std * std))


def masked_pose_tracking(
    env: ManagerBasedRlEnv,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Track only commanded axes, normalized by active mask cardinality.

    Environments with no active axes return zero instead of receiving a free
    constant reward. This is essential during the locomotion-only warm-up.
    """
    targets = _selected_command_axes(ballet_targets(env), asset_cfg)
    mask = _selected_command_axes(ballet_mask(env), asset_cfg)
    error = (joint_pos_normalized(env, asset_cfg) - targets).square()
    active_count = mask.sum(dim=1)
    mse = (error * mask).sum(dim=1) / active_count.clamp_min(1.0)
    score = torch.exp(-mse / (std * std))
    return torch.where(active_count > 0.0, score, torch.zeros_like(score))


def unmasked_home_tracking(
    env: ManagerBasedRlEnv,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Keep selected, non-commanded joints near the model's default pose."""
    robot = env.scene[asset_cfg.name]
    current = robot.data.joint_pos[:, asset_cfg.joint_ids]
    home = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    limits = robot.data.joint_pos_limits[:, asset_cfg.joint_ids]
    scale = (limits[..., 1] - limits[..., 0]).clamp_min(1.0e-6) * 0.5
    error = ((current - home) / scale).square()
    inactive_mask = 1.0 - _selected_command_axes(ballet_mask(env), asset_cfg)
    inactive_count = inactive_mask.sum(dim=1)
    mse = (error * inactive_mask).sum(dim=1) / inactive_count.clamp_min(1.0)
    score = torch.exp(-mse / (std * std))
    return torch.where(inactive_count > 0.0, score, torch.zeros_like(score))


def fell_over_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return 1 only when the fell_over termination has fired."""
    return env.termination_manager.get_term("fell_over").float()


def non_finite_state_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return one only for numerical-failure termination, never for timeout."""
    return env.termination_manager.get_term("non_finite_state").float()


def commanded_leg_ground_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    *,
    left_leg_cfg: SceneEntityCfg,
    right_leg_cfg: SceneEntityCfg,
    velocity_epsilon: float = VELOCITY_EPSILON,
) -> torch.Tensor:
    """Penalize invalid foot contacts using velocity- and mask-aware rules.

    For command speed below ``velocity_epsilon``:
    * no leg masks: both feet must contact the ground;
    * exactly one leg masked: the masked foot must be off the ground and the
      unmasked foot must contact the ground;
    * both legs masked: no contact rule is enforced.

    For command speed at or above ``velocity_epsilon``:
    * exactly one leg masked: the masked foot must be off the ground;
    * no leg masks or both legs masked: no contact rule is enforced.
    """
    sensor = env.scene[sensor_name]
    if not isinstance(sensor, ContactSensor):
        raise TypeError(f"{sensor_name!r} must be a ContactSensor, got {type(sensor).__name__}")
    if sensor.data.found is None:
        raise ValueError(f"ContactSensor {sensor_name!r} must expose the 'found' field")

    try:
        foot_ids = [
            sensor.primary_names.index("left_ankle_roll_link"),
            sensor.primary_names.index("right_ankle_roll_link"),
        ]
    except ValueError as exc:
        raise ValueError(
            f"ContactSensor {sensor_name!r} must contain G1 foot primaries; got {sensor.primary_names}"
        ) from exc

    contacts = sensor.data.found[:, foot_ids] > 0
    left_contact, right_contact = contacts[:, 0], contacts[:, 1]
    left_active = _group_has_active_mask(env, left_leg_cfg)
    right_active = _group_has_active_mask(env, right_leg_cfg)

    no_masks = ~left_active & ~right_active
    left_only = left_active & ~right_active
    right_only = right_active & ~left_active

    command_speed = env.command_manager.get_command("ballet")[:, :3].abs().amax(dim=1)
    stationary = command_speed < velocity_epsilon
    moving = ~stationary

    penalty = torch.zeros_like(command_speed)
    penalty += (stationary & no_masks).float() * ((~left_contact).float() + (~right_contact).float())
    penalty += (stationary & left_only).float() * (left_contact.float() + (~right_contact).float())
    penalty += (stationary & right_only).float() * (right_contact.float() + (~left_contact).float())
    penalty += (moving & left_only & left_contact).float()
    penalty += (moving & right_only & right_contact).float()
    return penalty


def com_support_projection_tracking(
    env: ManagerBasedRlEnv,
    std: float = 0.12,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    *,
    feet_cfg: SceneEntityCfg,
    left_leg_cfg: SceneEntityCfg,
    right_leg_cfg: SceneEntityCfg,
    velocity_epsilon: float = VELOCITY_EPSILON,
) -> torch.Tensor:
    """Track CoM support projection only when locomotion/support control needs it.

    The reward is active when command speed is at or above ``velocity_epsilon``.
    Below the threshold it is active only when exactly one leg has an active
    mask. Otherwise it returns zero.
    """
    command_speed = env.command_manager.get_command("ballet")[:, :3].abs().amax(dim=1)
    stationary = command_speed < velocity_epsilon
    left_active = _group_has_active_mask(env, left_leg_cfg)
    right_active = _group_has_active_mask(env, right_leg_cfg)
    exactly_one_leg_masked = left_active ^ right_active
    active = ~stationary | (stationary & exactly_one_leg_masked)

    com_xy = whole_body_com_xy_b(env, asset_cfg)
    support_xy = support_center_xy_b(
        env,
        asset_cfg=asset_cfg,
        feet_cfg=feet_cfg,
        left_leg_cfg=left_leg_cfg,
        right_leg_cfg=right_leg_cfg,
    )
    squared_distance = (com_xy - support_xy).square().sum(dim=1)
    score = torch.exp(-squared_distance / (std * std))
    return torch.where(active, score, torch.zeros_like(score))

def pelvis_height_tracking(
    env: ManagerBasedRlEnv,
    target_height: float = 0.8,
    std: float = 0.18,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    *,
    left_leg_cfg: SceneEntityCfg,
    right_leg_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Keep standing pelvis height when no leg has an explicit ballet target."""
    robot = env.scene[asset_cfg.name]
    height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    error = height - target_height
    score = torch.exp(-error.square() / (std * std))
    any_leg_commanded = _group_has_active_mask(env, left_leg_cfg) | _group_has_active_mask(
        env, right_leg_cfg
    )
    return torch.where(any_leg_commanded, torch.zeros_like(score), score)


def unmasked_leg_lateral_alignment(
    env: ManagerBasedRlEnv,
    std: float = 0.35,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    *,
    left_leg_cfg: SceneEntityCfg,
    right_leg_cfg: SceneEntityCfg,
    left_alignment_cfg: SceneEntityCfg,
    right_alignment_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Keep uncommanded legs near a human sagittal-plane alignment.

    Only hip roll/yaw and ankle roll are regularized. Hip/knee/ankle pitch
    remain free to produce a walking cycle. An active mask anywhere on one
    leg disables this prior only for that leg.
    """
    robot = env.scene[asset_cfg.name]

    def side_score(cfg: SceneEntityCfg) -> torch.Tensor:
        current = robot.data.joint_pos[:, cfg.joint_ids]
        home = robot.data.default_joint_pos[:, cfg.joint_ids]
        limits = robot.data.joint_pos_limits[:, cfg.joint_ids]
        half_range = (0.5 * (limits[..., 1] - limits[..., 0])).clamp_min(1.0e-6)
        mse = ((current - home) / half_range).square().mean(dim=1)
        return torch.exp(-mse / (std * std))

    left_eligible = ~_group_has_active_mask(env, left_leg_cfg)
    right_eligible = ~_group_has_active_mask(env, right_leg_cfg)
    eligible_count = left_eligible.float() + right_eligible.float()
    score = (
        side_score(left_alignment_cfg) * left_eligible.float()
        + side_score(right_alignment_cfg) * right_eligible.float()
    ) / eligible_count.clamp_min(1.0)
    return torch.where(eligible_count > 0.0, score, torch.zeros_like(score))


def unmasked_foot_heading_alignment(
    env: ManagerBasedRlEnv,
    std: float = 0.45,
    turn_rate_threshold: float = 0.15,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    *,
    feet_cfg: SceneEntityCfg,
    left_leg_cfg: SceneEntityCfg,
    right_leg_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Align uncommanded feet with the base heading during near-straight motion."""
    from mjlab.utils.lab_api.math import quat_apply

    robot = env.scene[asset_cfg.name]
    foot_quat = robot.data.site_quat_w[:, feet_cfg.site_ids]
    if foot_quat.shape[1] != 2:
        raise ValueError("unmasked_foot_heading_alignment expects ordered left/right foot sites")

    forward_local = torch.zeros_like(foot_quat[..., :3])
    forward_local[..., 0] = 1.0
    foot_forward = quat_apply(foot_quat, forward_local)[..., :2]
    foot_forward = foot_forward / foot_forward.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)

    heading = robot.data.heading_w
    base_forward = torch.stack((torch.cos(heading), torch.sin(heading)), dim=1)
    dot = (foot_forward * base_forward[:, None, :]).sum(dim=2).clamp(-1.0, 1.0)
    cross = (
        base_forward[:, None, 0] * foot_forward[:, :, 1]
        - base_forward[:, None, 1] * foot_forward[:, :, 0]
    )
    heading_error = torch.atan2(cross, dot)
    side_scores = torch.exp(-heading_error.square() / (std * std))

    eligible = torch.stack(
        (
            ~_group_has_active_mask(env, left_leg_cfg),
            ~_group_has_active_mask(env, right_leg_cfg),
        ),
        dim=1,
    )
    straight = env.command_manager.get_command("ballet")[:, 2].abs() <= turn_rate_threshold
    eligible = eligible & straight[:, None]
    eligible_count = eligible.sum(dim=1)
    score = (side_scores * eligible.float()).sum(dim=1) / eligible_count.clamp_min(1)
    return torch.where(eligible_count > 0, score, torch.zeros_like(score))
