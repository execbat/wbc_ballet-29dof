"""Curriculum terms specific to the ballet command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .commands import BalletCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _linear_schedule_at_step(
    step: int,
    *,
    warmup_steps: int,
    ramp_steps: int,
    initial_value: float,
    final_value: float,
) -> tuple[float, float]:
    """Return ``(value, progress)`` for a clipped linear schedule."""
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if ramp_steps <= 0:
        raise ValueError("ramp_steps must be positive")
    if initial_value > final_value:
        raise ValueError("initial_value must not exceed final_value")

    progress = min(max((step - warmup_steps) / ramp_steps, 0.0), 1.0)
    value = initial_value + progress * (final_value - initial_value)
    return value, progress


def mask_probability_at_step(
    step: int,
    *,
    warmup_steps: int,
    ramp_steps: int,
    initial_probability: float,
    final_probability: float,
) -> tuple[float, float]:
    """Return ``(probability, progress)`` for a clipped linear schedule."""
    if not 0.0 <= initial_probability <= final_probability <= 1.0:
        raise ValueError("probabilities must satisfy 0 <= initial <= final <= 1")
    return _linear_schedule_at_step(
        step,
        warmup_steps=warmup_steps,
        ramp_steps=ramp_steps,
        initial_value=initial_probability,
        final_value=final_probability,
    )


def target_scale_at_step(
    step: int,
    *,
    warmup_steps: int,
    ramp_steps: int,
    initial_scale: float,
    final_scale: float,
) -> tuple[float, float]:
    """Return the normalized joint-target scale and curriculum progress."""
    if not 0.0 <= initial_scale <= final_scale <= 1.0:
        raise ValueError("target scales must satisfy 0 <= initial <= final <= 1")
    return _linear_schedule_at_step(
        step,
        warmup_steps=warmup_steps,
        ramp_steps=ramp_steps,
        initial_value=initial_scale,
        final_value=final_scale,
    )


def mask_probability_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    *,
    command_name: str,
    warmup_steps: int,
    ramp_steps: int,
    initial_probability: float = 0.0,
    final_probability: float = 0.15,
    num_joints: int = 29,
) -> dict[str, torch.Tensor]:
    """Keep masks disabled, then linearly introduce masked joint targets.

    MJLab evaluates curriculum terms before resetting/resampling commands, so
    changing the command term cfg here applies to the command sampled for the
    new episode. ``common_step_counter`` counts policy/environment steps.
    """
    del env_ids  # The schedule is global and deterministic across environments.
    probability, progress = mask_probability_at_step(
        env.common_step_counter,
        warmup_steps=warmup_steps,
        ramp_steps=ramp_steps,
        initial_probability=initial_probability,
        final_probability=final_probability,
    )

    command = env.command_manager.get_term(command_name)
    if not isinstance(command, BalletCommand):
        raise TypeError(f"command {command_name!r} must be a BalletCommand")
    command.cfg.mask_probability = probability

    return {
        "probability": torch.tensor(probability, device=env.device),
        "progress": torch.tensor(progress, device=env.device),
        "expected_active_axes": torch.tensor(probability * num_joints, device=env.device),
    }


def target_scale_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    *,
    command_name: str,
    warmup_steps: int,
    ramp_steps: int,
    initial_scale: float = 0.0,
    final_scale: float = 0.8,
) -> dict[str, torch.Tensor]:
    """Keep targets at zero, then linearly grow their normalized range."""
    del env_ids  # The schedule is global and deterministic across environments.
    scale, progress = target_scale_at_step(
        env.common_step_counter,
        warmup_steps=warmup_steps,
        ramp_steps=ramp_steps,
        initial_scale=initial_scale,
        final_scale=final_scale,
    )

    command = env.command_manager.get_term(command_name)
    if not isinstance(command, BalletCommand):
        raise TypeError(f"command {command_name!r} must be a BalletCommand")
    command.cfg.target_scale = scale

    return {
        "scale": torch.tensor(scale, device=env.device),
        "progress": torch.tensor(progress, device=env.device),
    }
