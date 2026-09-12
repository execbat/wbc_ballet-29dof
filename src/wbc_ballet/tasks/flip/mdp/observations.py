"""Observation helpers owned by the flip task."""

from __future__ import annotations

import torch
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

_ROBOT_CFG = SceneEntityCfg("robot")
_NUM_JOINTS = 29
_TARGET_START = 3
_TARGET_END = _TARGET_START + _NUM_JOINTS
_MASK_START = _TARGET_END
_MASK_END = _MASK_START + _NUM_JOINTS


def _command(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.command_manager.get_command("ballet")


def ballet_targets(env):
    return _command(env)[:, _TARGET_START:_TARGET_END]


def ballet_mask(env):
    return _command(env)[:, _MASK_START:_MASK_END]


def masked_ballet_targets(env):
    targets = ballet_targets(env)
    return torch.where(ballet_mask(env) >= 0.5, targets, torch.zeros_like(targets))


def ballet_velocity(env):
    return _command(env)[:, :3]


def joint_pos_normalized(env, asset_cfg: SceneEntityCfg = _ROBOT_CFG):
    robot = env.scene[asset_cfg.name]
    positions = robot.data.joint_pos[:, asset_cfg.joint_ids]
    limits = robot.data.joint_pos_limits[:, asset_cfg.joint_ids]
    lower, upper = limits[..., 0], limits[..., 1]
    return (2.0 * (positions - lower) / (upper - lower).clamp_min(1.0e-6) - 1.0).clamp(-1.0, 1.0)


def _whole_body_com_pos_w(robot):
    return robot.data.data.subtree_com[:, robot.data.indexing.root_body_id]


def _world_xy_to_base_heading(robot, point_xy_w):
    delta = point_xy_w - robot.data.root_link_pos_w[:, :2]
    heading = robot.data.heading_w
    c, s = torch.cos(heading), torch.sin(heading)
    return torch.stack((c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]), dim=1)


def whole_body_com_xy_b(env, asset_cfg: SceneEntityCfg = _ROBOT_CFG):
    robot = env.scene[asset_cfg.name]
    return _world_xy_to_base_heading(robot, _whole_body_com_pos_w(robot)[:, :2])


def _group_has_active_mask(env, asset_cfg: SceneEntityCfg):
    return (ballet_mask(env)[:, asset_cfg.joint_ids] >= 0.5).any(dim=1)


def support_center_xy_b(
    env,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    *,
    feet_cfg: SceneEntityCfg,
    left_leg_cfg: SceneEntityCfg,
    right_leg_cfg: SceneEntityCfg,
):
    robot = env.scene[asset_cfg.name]
    points = robot.data.site_pos_w[:, feet_cfg.site_ids, :2]
    if points.shape[1] != 2:
        raise ValueError("support_center_xy_b expects two ordered support sites")
    left_active = _group_has_active_mask(env, left_leg_cfg)
    right_active = _group_has_active_mask(env, right_leg_cfg)
    left, right = points[:, 0], points[:, 1]
    center = torch.where(
        (left_active & ~right_active)[:, None],
        right,
        torch.where((right_active & ~left_active)[:, None], left, 0.5 * (left + right)),
    )
    return _world_xy_to_base_heading(robot, center)
