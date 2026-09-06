"""Termination terms specific to the ballet task."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.nan_guard import NanGuard

_ROBOT_CFG = SceneEntityCfg("robot")


def pelvis_height_below(
    env: ManagerBasedRlEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Terminate when the pelvis is below ``minimum_height`` above its env origin.

    The G1 free/root body is ``pelvis``. Subtracting the environment origin
    keeps the 0.2 m threshold meaningful on generated terrain whose world-frame
    spawn height is not necessarily zero.
    """
    robot = env.scene[asset_cfg.name]
    pelvis_height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return pelvis_height < minimum_height


def non_finite_state_or_action(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Terminate only envs containing invalid physics state or policy actions.

    Physics failures are detected before reset, while the observation manager
    separately sanitizes the post-reset tensors returned to RSL-RL.
    """
    invalid_state = NanGuard.detect_nans(env.sim.data)
    actions = env.action_manager.action
    invalid_action = (~torch.isfinite(actions)).reshape(actions.shape[0], -1).any(dim=1)
    return invalid_state | invalid_action
