"""Finite-value diagnostics exported through MJLab's metrics manager."""

from __future__ import annotations

from typing import Literal

import torch
from mjlab.envs import ManagerBasedRlEnv

_PHYSICS_FIELDS = ("qpos", "qvel", "qacc", "qacc_warmstart", "sensordata")


def _per_environment_any(values: torch.Tensor) -> torch.Tensor:
    """Reduce every non-batch dimension without assuming a fixed tensor rank."""
    return values.reshape(values.shape[0], -1).any(dim=1)


def nonfinite_physics_component(
    env: ManagerBasedRlEnv,
    component: Literal["qpos", "qvel", "qacc", "qacc_warmstart", "sensordata"],
) -> torch.Tensor:
    """Return one for envs whose selected MuJoCo state component is not finite."""
    values = getattr(env.sim.data, component)
    return _per_environment_any(~torch.isfinite(values)).float()


def invalid_physics_value_type(
    env: ManagerBasedRlEnv,
    value_type: Literal["nan", "inf"],
) -> torch.Tensor:
    """Return one when any physics-state component contains NaN or Inf."""
    predicate = torch.isnan if value_type == "nan" else torch.isinf
    invalid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for component in _PHYSICS_FIELDS:
        invalid |= _per_environment_any(predicate(getattr(env.sim.data, component)))
    return invalid.float()


def nonfinite_policy_action(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return one when the current raw policy action contains NaN or Inf."""
    return _per_environment_any(~torch.isfinite(env.action_manager.action)).float()
