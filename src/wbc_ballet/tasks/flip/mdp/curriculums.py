"""Curriculum terms owned by the flip task."""
import torch
from .commands import BalletCommand


def _linear_schedule_at_step(step, *, warmup_steps, ramp_steps, initial_value, final_value):
    if warmup_steps < 0 or ramp_steps <= 0 or initial_value > final_value:
        raise ValueError("invalid linear curriculum parameters")
    progress = min(max((step - warmup_steps) / ramp_steps, 0.0), 1.0)
    return initial_value + progress * (final_value - initial_value), progress


def mask_probability_curriculum(env, env_ids, *, command_name, warmup_steps, ramp_steps, initial_probability=.0, final_probability=.15, num_joints=29):
    del env_ids
    if not 0 <= initial_probability <= final_probability <= 1:
        raise ValueError("probabilities must satisfy 0 <= initial <= final <= 1")
    probability, progress = _linear_schedule_at_step(env.common_step_counter, warmup_steps=warmup_steps, ramp_steps=ramp_steps, initial_value=initial_probability, final_value=final_probability)
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, BalletCommand):
        raise TypeError(f"command {command_name!r} must be a BalletCommand")
    command.cfg.mask_probability = probability
    return {"probability": torch.tensor(probability, device=env.device), "progress": torch.tensor(progress, device=env.device), "expected_active_axes": torch.tensor(probability * num_joints, device=env.device)}


def target_scale_curriculum(env, env_ids, *, command_name, warmup_steps, ramp_steps, initial_scale=.0, final_scale=.8):
    del env_ids
    if not 0 <= initial_scale <= final_scale <= 1:
        raise ValueError("target scales must satisfy 0 <= initial <= final <= 1")
    scale, progress = _linear_schedule_at_step(env.common_step_counter, warmup_steps=warmup_steps, ramp_steps=ramp_steps, initial_value=initial_scale, final_value=final_scale)
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, BalletCommand):
        raise TypeError(f"command {command_name!r} must be a BalletCommand")
    command.cfg.target_scale = scale
    return {"scale": torch.tensor(scale, device=env.device), "progress": torch.tensor(progress, device=env.device)}


def staged_value_at_step(step, *, start_steps, stage_interval_steps, stage_values):
    if start_steps < 0 or stage_interval_steps <= 0 or not stage_values:
        raise ValueError("invalid staged curriculum parameters")
    if step < start_steps:
        return 0.0, -1
    stage = min((step - start_steps) // stage_interval_steps, len(stage_values) - 1)
    return float(stage_values[stage]), int(stage)


def reward_weight_curriculum(env, env_ids, *, reward_name, start_steps, stage_interval_steps, stage_weights):
    del env_ids
    weight, stage = staged_value_at_step(env.common_step_counter, start_steps=start_steps, stage_interval_steps=stage_interval_steps, stage_values=stage_weights)
    env.reward_manager.get_term_cfg(reward_name).weight = weight
    return {"weight": torch.tensor(weight, device=env.device), "stage": torch.tensor(stage, device=env.device)}


def flip_probability_curriculum(
    env, env_ids, *,
    warmup_steps=0,
    ramp_steps,
    initial_probability=1.0,
    final_probability=0.5,
):
    """Linearly ramp ``reset_robot``'s ``flip_probability`` (chance a reset
    spawns handstand vs. standing).

    Starts fully handstand (every reset spawns flip=1) so early training
    only ever has to work on the hand-support half of the task, then ramps
    down toward an even draw as training progresses. Same schedule shape as
    ballet's ``mask_probability_curriculum``, just decreasing instead of
    increasing, and targeting an event term's params instead of a command's.
    """
    del env_ids  # The schedule is global and deterministic across environments.
    if not 0.0 <= final_probability <= initial_probability <= 1.0:
        raise ValueError("probabilities must satisfy 0 <= final <= initial <= 1")
    step = env.common_step_counter
    progress = min(max((step - warmup_steps) / ramp_steps, 0.0), 1.0)
    probability = initial_probability + progress * (final_probability - initial_probability)
    env.event_manager.get_term_cfg("reset_robot").params["flip_probability"] = probability
    return {
        "probability": torch.tensor(probability, device=env.device),
        "progress": torch.tensor(progress, device=env.device),
    }
