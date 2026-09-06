from types import SimpleNamespace

import pytest
import torch

from wbc_ballet.mdp.commands import BalletCommandCfg, blend_joint_targets
from wbc_ballet.mdp.curriculums import mask_probability_at_step, target_scale_at_step
from wbc_ballet.tasks.ballet.ballet_curriculum_cfg import (
    FINAL_MASK_PROBABILITY,
    FINAL_TARGET_SCALE,
    MASK_RAMP_ITERATIONS,
    PPO_STEPS_PER_ITERATION,
    TARGET_SCALE_RAMP_ITERATIONS,
    WALK_ONLY_ITERATIONS,
)


def test_mask_probability_schedule_boundaries() -> None:
    warmup = WALK_ONLY_ITERATIONS * PPO_STEPS_PER_ITERATION
    ramp = MASK_RAMP_ITERATIONS * PPO_STEPS_PER_ITERATION
    kwargs = {
        "warmup_steps": warmup,
        "ramp_steps": ramp,
        "initial_probability": 0.0,
        "final_probability": FINAL_MASK_PROBABILITY,
    }

    assert mask_probability_at_step(0, **kwargs) == (0.0, 0.0)
    assert mask_probability_at_step(warmup, **kwargs) == (0.0, 0.0)
    probability, progress = mask_probability_at_step(warmup + ramp // 2, **kwargs)
    assert progress == pytest.approx(0.5)
    assert probability == pytest.approx(FINAL_MASK_PROBABILITY / 2.0)
    assert mask_probability_at_step(warmup + ramp, **kwargs) == (
        FINAL_MASK_PROBABILITY,
        1.0,
    )
    assert mask_probability_at_step(warmup + 10 * ramp, **kwargs) == (
        FINAL_MASK_PROBABILITY,
        1.0,
    )


def test_target_scale_schedule_boundaries() -> None:
    warmup = WALK_ONLY_ITERATIONS * PPO_STEPS_PER_ITERATION
    ramp = TARGET_SCALE_RAMP_ITERATIONS * PPO_STEPS_PER_ITERATION
    kwargs = {
        "warmup_steps": warmup,
        "ramp_steps": ramp,
        "initial_scale": 0.0,
        "final_scale": FINAL_TARGET_SCALE,
    }

    assert target_scale_at_step(0, **kwargs) == (0.0, 0.0)
    assert target_scale_at_step(warmup, **kwargs) == (0.0, 0.0)
    scale, progress = target_scale_at_step(warmup + ramp // 2, **kwargs)
    assert progress == pytest.approx(0.5)
    assert scale == pytest.approx(FINAL_TARGET_SCALE / 2.0)
    assert target_scale_at_step(warmup + ramp, **kwargs) == (
        FINAL_TARGET_SCALE,
        1.0,
    )


def test_target_scale_blends_from_current_pose_to_full_goal() -> None:
    current = torch.tensor([[-0.739, 0.2, 0.7]])
    goal = torch.tensor([[0.8, -0.8, 0.0]])

    torch.testing.assert_close(
        blend_joint_targets(current, goal, target_scale=0.0, target_limit=0.8),
        current,
    )
    torch.testing.assert_close(
        blend_joint_targets(current, goal, target_scale=0.4, target_limit=0.8),
        0.5 * (current + goal),
    )
    torch.testing.assert_close(
        blend_joint_targets(current, goal, target_scale=0.8, target_limit=0.8),
        goal,
    )


def test_zero_target_scale_resamples_the_current_normalized_pose() -> None:
    current = torch.linspace(-0.75, 0.75, 29).repeat(2, 1)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=current.clone(),
            joint_pos_limits=torch.tensor([-1.0, 1.0]).repeat(2, 29, 1),
        )
    )
    env = SimpleNamespace(num_envs=2, device="cpu", scene={"robot": robot})
    cfg = BalletCommandCfg(
        resampling_time_range=(6.0, 10.0),
        target_scale=0.0,
        target_limit=0.8,
        mask_probability=1.0,
    )
    command = cfg.build(env)

    command._resample_command(torch.tensor([0, 1]))

    torch.testing.assert_close(command.command[:, 3:32], current)
