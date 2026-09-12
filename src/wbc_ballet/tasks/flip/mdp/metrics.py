"""Finite-value diagnostic metrics."""

import torch

_PHYSICS_FIELDS = ("qpos", "qvel", "qacc", "qacc_warmstart", "sensordata")


def _any(values):
    return values.reshape(values.shape[0], -1).any(1)


def nonfinite_physics_component(env, component):
    return _any(~torch.isfinite(getattr(env.sim.data, component))).float()


def invalid_physics_value_type(env, value_type):
    predicate = torch.isnan if value_type == "nan" else torch.isinf
    invalid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for component in _PHYSICS_FIELDS:
        invalid |= _any(predicate(getattr(env.sim.data, component)))
    return invalid.float()


def nonfinite_policy_action(env):
    return _any(~torch.isfinite(env.action_manager.action)).float()
