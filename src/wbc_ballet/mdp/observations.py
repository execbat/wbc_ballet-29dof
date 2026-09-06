from __future__ import annotations

import torch
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

_ROBOT_CFG = SceneEntityCfg("robot")

# Must match the layout in ``wbc_ballet.mdp.commands``:
# [velocity(3), targets(29), mask(29)] = 61D. 29 = the G1-29DoF joint count.
_NUM_JOINTS = 29
_TARGET_START = 3
_TARGET_END = _TARGET_START + _NUM_JOINTS  # 32
_MASK_START = _TARGET_END
_MASK_END = _MASK_START + _NUM_JOINTS  # 61


def _command(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.command_manager.get_command("ballet")


def ballet_targets(env: ManagerBasedRlEnv) -> torch.Tensor:
    return _command(env)[:, _TARGET_START:_TARGET_END]


def ballet_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
    return _command(env)[:, _MASK_START:_MASK_END]


def masked_ballet_targets(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return policy-visible targets, with every inactive axis forced to zero."""
    targets = ballet_targets(env)
    return torch.where(ballet_mask(env) >= 0.5, targets, torch.zeros_like(targets))


def ballet_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
    return _command(env)[:, :3]


def joint_pos_normalized(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT_CFG
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    positions = robot.data.joint_pos[:, asset_cfg.joint_ids]
    limits = robot.data.joint_pos_limits[:, asset_cfg.joint_ids]
    lower, upper = limits[..., 0], limits[..., 1]
    return (2.0 * (positions - lower) / (upper - lower).clamp_min(1.0e-6) - 1.0).clamp(-1, 1)


def _whole_body_com_pos_w(robot) -> torch.Tensor:
    """Return the MuJoCo subtree CoM for the complete articulated robot."""
    root_body_id = robot.data.indexing.root_body_id
    return robot.data.data.subtree_com[:, root_body_id]


def _world_xy_to_base_heading(robot, point_xy_w: torch.Tensor) -> torch.Tensor:
    """Express a world XY point relative to the base in its yaw-only frame."""
    delta = point_xy_w - robot.data.root_link_pos_w[:, :2]
    heading = robot.data.heading_w
    cos_heading = torch.cos(heading)
    sin_heading = torch.sin(heading)
    return torch.stack(
        (
            cos_heading * delta[:, 0] + sin_heading * delta[:, 1],
            -sin_heading * delta[:, 0] + cos_heading * delta[:, 1],
        ),
        dim=1,
    )


def whole_body_com_xy_b(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT_CFG
) -> torch.Tensor:
    """Ground projection of the whole-robot CoM relative to the base, in base yaw frame."""
    robot = env.scene[asset_cfg.name]
    return _world_xy_to_base_heading(robot, _whole_body_com_pos_w(robot)[:, :2])


def _group_has_active_mask(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    selected = ballet_mask(env)[:, asset_cfg.joint_ids]
    return (selected >= 0.5).any(dim=1)


def support_center_xy_b(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    *,
    feet_cfg: SceneEntityCfg,
    left_leg_cfg: SceneEntityCfg,
    right_leg_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Mask-aware center of the support feet, relative to the base in XY.

    With masks on exactly one leg, the opposite leg is the sole support. With
    masks on neither leg or on both legs, both feet define the support center.
    This intentionally follows command semantics rather than contact state.
    """
    robot = env.scene[asset_cfg.name]
    feet_xy_w = robot.data.site_pos_w[:, feet_cfg.site_ids, :2]
    if feet_xy_w.shape[1] != 2:
        raise ValueError("support_center_xy_b expects exactly two ordered foot sites: left, right")

    left_active = _group_has_active_mask(env, left_leg_cfg)
    right_active = _group_has_active_mask(env, right_leg_cfg)
    left_foot = feet_xy_w[:, 0]
    right_foot = feet_xy_w[:, 1]
    midpoint = 0.5 * (left_foot + right_foot)

    only_left_commanded = left_active & ~right_active
    only_right_commanded = right_active & ~left_active
    support_xy_w = torch.where(
        only_left_commanded[:, None],
        right_foot,
        torch.where(only_right_commanded[:, None], left_foot, midpoint),
    )
    return _world_xy_to_base_heading(robot, support_xy_w)
