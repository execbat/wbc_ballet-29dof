"""Diagnostic event terms for the ballet task."""

from __future__ import annotations

import torch
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .observations import ballet_mask, joint_pos_normalized, masked_ballet_targets

_ROBOT_CFG = SceneEntityCfg("robot")


def _as_joint_id_list(joint_ids: list[int] | slice, joint_count: int) -> list[int]:
    if isinstance(joint_ids, slice):
        return list(range(joint_count))[joint_ids]
    return list(joint_ids)


def print_joint_observation_table(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    *,
    env_index: int = 0,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> None:
    """Print absolute/normalized joint state and the exact target seen by the actor.

    This event is configured as a global interval term, so ``env_ids`` is not
    used. Only one environment and four 29-element vectors are copied to CPU.
    """
    del env_ids
    if not 0 <= env_index < env.num_envs:
        raise IndexError(f"env_index={env_index} is outside [0, {env.num_envs})")

    robot = env.scene[asset_cfg.name]
    joint_ids = _as_joint_id_list(asset_cfg.joint_ids, len(robot.joint_names))
    joint_names = [robot.joint_names[joint_id] for joint_id in joint_ids]

    absolute = robot.data.joint_pos[env_index, joint_ids].detach().cpu().tolist()
    home = robot.data.default_joint_pos[env_index, joint_ids].detach().cpu().tolist()
    normalized = joint_pos_normalized(env, asset_cfg)[env_index].detach().cpu().tolist()
    mask = ballet_mask(env)[env_index, joint_ids].detach().cpu().tolist()
    target_obs = masked_ballet_targets(env)[env_index, joint_ids].detach().cpu().tolist()

    lines = [
        f"\n[G1 joint observation | env={env_index} | step={env.common_step_counter}]",
        (
            f"{'#':>2}  {'joint':34} {'q_actual':>10} {'q_home':>9} {'delta':>9} "
            f"{'q_norm':>9} {'mask':>5} {'target_obs':>10}"
        ),
        "-" * 99,
    ]
    for index, (name, q_abs, q_home, q_norm, enabled, target) in enumerate(
        zip(joint_names, absolute, home, normalized, mask, target_obs, strict=True)
    ):
        lines.append(
            f"{index:2d}  {name:34} {q_abs:10.4f} {q_home:9.4f} "
            f"{q_abs - q_home:9.4f} {q_norm:9.4f} "
            f"{enabled:5.0f} {target:10.4f}"
        )
    print("\n".join(lines), flush=True)
