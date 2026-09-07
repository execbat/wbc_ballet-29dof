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
    velocity_epsilon: float = 1.0e-4,
) -> torch.Tensor:
    """Penalize missing ground contact for the support leg(s) while stationary.

    The term is active only when all three ``velocity_commands`` are zero
    (within ``velocity_epsilon``). Support-leg semantics are mask-aware:

    * no active leg masks -> both feet are support feet and both must contact ground;
    * exactly one leg masked -> the other, unmasked leg is the support leg;
    * both legs masked -> the leg whose mask became active first is treated as
      the commanded/non-supporting leg, and the other leg is the support leg.

    If both leg masks rise on the same control step there is no unique
    "earlier" leg, so both feet are conservatively required to stay in contact.

    Returns the number of required support feet that are not touching ground
    (0, 1, or 2). The reward configuration applies a negative weight.
    """
    sensor = env.scene[sensor_name]
    if not isinstance(sensor, ContactSensor):
        raise TypeError(f"{sensor_name!r} must be a ContactSensor, got {type(sensor).__name__}")
    found = sensor.data.found
    if found is None:
        raise ValueError(f"ContactSensor {sensor_name!r} must expose the 'found' field")

    primary_names = sensor.primary_names
    try:
        foot_indices = [
            primary_names.index("left_ankle_roll_link"),
            primary_names.index("right_ankle_roll_link"),
        ]
    except ValueError as exc:
        raise ValueError(
            f"ContactSensor {sensor_name!r} must contain G1 foot primaries; got {primary_names}"
        ) from exc

    contacts = found[:, foot_indices] > 0
    left_active = _group_has_active_mask(env, left_leg_cfg)
    right_active = _group_has_active_mask(env, right_leg_cfg)

    # Track rising-edge activation time per environment. This state is updated
    # even while the robot is moving so support selection is correct when the
    # commanded velocity later returns to zero.
    state_name = "_commanded_leg_ground_contact_state"
    state = getattr(env, state_name, None)
    num_envs = left_active.shape[0]
    if state is None or state["left_prev"].shape[0] != num_envs:
        inf = torch.full((num_envs,), float("inf"), device=left_active.device)
        state = {
            "left_prev": torch.zeros_like(left_active),
            "right_prev": torch.zeros_like(right_active),
            "left_on_step": inf.clone(),
            "right_on_step": inf.clone(),
        }
        setattr(env, state_name, state)

    step = float(getattr(env, "common_step_counter", 0))
    left_rising = left_active & ~state["left_prev"]
    right_rising = right_active & ~state["right_prev"]
    state["left_on_step"] = torch.where(
        left_rising, torch.full_like(state["left_on_step"], step), state["left_on_step"]
    )
    state["right_on_step"] = torch.where(
        right_rising, torch.full_like(state["right_on_step"], step), state["right_on_step"]
    )
    state["left_on_step"] = torch.where(
        left_active, state["left_on_step"], torch.full_like(state["left_on_step"], float("inf"))
    )
    state["right_on_step"] = torch.where(
        right_active, state["right_on_step"], torch.full_like(state["right_on_step"], float("inf"))
    )
    state["left_prev"] = left_active.clone()
    state["right_prev"] = right_active.clone()

    support_left = ~left_active & ~right_active
    support_right = support_left.clone()

    # One masked leg: the unmasked leg is the support leg.
    support_left |= ~left_active & right_active
    support_right |= left_active & ~right_active

    # Both masked: earlier activation is non-supporting; later activation is
    # support. Equal activation times are ambiguous, so require both contacts.
    both_active = left_active & right_active
    left_earlier = state["left_on_step"] < state["right_on_step"]
    right_earlier = state["right_on_step"] < state["left_on_step"]
    simultaneous = both_active & ~(left_earlier | right_earlier)
    support_right |= both_active & left_earlier
    support_left |= both_active & right_earlier
    support_left |= simultaneous
    support_right |= simultaneous

    command = env.command_manager.get_command("ballet")[:, :3]
    stationary = command.abs().amax(dim=1) <= velocity_epsilon
    missing_support_contact = torch.stack(
        (support_left & ~contacts[:, 0], support_right & ~contacts[:, 1]), dim=1
    )
    return torch.where(
        stationary,
        missing_support_contact.sum(dim=1).float(),
        torch.zeros(num_envs, device=contacts.device),
    )

def com_support_projection_tracking(
    env: ManagerBasedRlEnv,
    std: float = 0.12,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    *,
    feet_cfg: SceneEntityCfg,
    left_leg_cfg: SceneEntityCfg,
    right_leg_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Track the mask-aware support center with the whole-body CoM projection."""
    com_xy = whole_body_com_xy_b(env, asset_cfg)
    support_xy = support_center_xy_b(
        env,
        asset_cfg=asset_cfg,
        feet_cfg=feet_cfg,
        left_leg_cfg=left_leg_cfg,
        right_leg_cfg=right_leg_cfg,
    )
    squared_distance = (com_xy - support_xy).square().sum(dim=1)
    return torch.exp(-squared_distance / (std * std))


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
