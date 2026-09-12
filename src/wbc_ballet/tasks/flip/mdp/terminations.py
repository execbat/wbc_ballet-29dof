"""Termination terms owned by the flip task."""

import torch
from mjlab.utils.nan_guard import NanGuard


def non_finite_state_or_action(env):
    invalid_state = NanGuard.detect_nans(env.sim.data)
    actions = env.action_manager.action
    return invalid_state | (~torch.isfinite(actions)).reshape(actions.shape[0], -1).any(1)


def pelvis_height_below_for_duration(
    env,
    minimum_height: float = .2,
    duration_s: float = 2.,
    recovery_height: float = .35,
    recovery_duration_s: float = 2.,
):
    """Accumulate low time; clear it only after uninterrupted valid recovery."""
    if not hasattr(env, "_low_pelvis_time"):
        env._low_pelvis_time = torch.zeros(env.num_envs, device=env.device)
        env._pelvis_recovery_time = torch.zeros(env.num_envs, device=env.device)
    low_time = env._low_pelvis_time
    recovery_time = env._pelvis_recovery_time
    steps = getattr(env, "episode_length_buf", None)
    if steps is not None:
        new = steps <= 1
        low_time[new] = 0.
        recovery_time[new] = 0.
    height = env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    low = height < minimum_height
    low_time.add_(low.float() * float(env.step_dt))

    def contact(name):
        return (env.scene[name].data.found > 0).reshape(env.num_envs, -1).any(1)

    inverted = env.command_manager.get_command("flip")[:, 0] >= .5
    feet, hands = contact("feet_ground_contact"), contact("hands_ground_contact")
    correct = torch.where(inverted, hands, feet)
    wrong = torch.where(inverted, feet, hands)
    valid = (height > recovery_height) & correct & ~wrong & ~contact("head_ground_contact") & ~contact("forbidden_ground_contact")
    recovery_time[:] = torch.where(valid, recovery_time + float(env.step_dt), torch.zeros_like(recovery_time))
    recovered = recovery_time >= recovery_duration_s
    low_time[recovered] = 0.
    recovery_time[recovered] = 0.
    return low_time >= duration_s


def forbidden_contact_count_exceeded(
    env,
    sensor_name: str = "forbidden_ground_contact",
    maximum_contacts: int = 1,
    contact_interval_s: float = 0.5,
    grace_s: float = 0.0,
) -> torch.Tensor:
    """Terminate after the forbidden-contact counter exceeds its limit.

    A sustained forbidden contact increments the common counter every
    contact_interval_s. Therefore lying continuously on a forbidden body
    cannot be treated as only one contact.
    """
    num_envs = env.num_envs
    device = env.device
    dt = float(env.step_dt)

    found = env.scene[sensor_name].data.found
    if found is None:
        raise RuntimeError(
            f"Contact sensor {sensor_name!r} does not expose 'found'"
        )

    forbidden_contact = (
        found > 0
    ).reshape(num_envs, -1).any(dim=1)

    if not hasattr(env, "_forbidden_contact_count"):
        env._forbidden_contact_count = torch.zeros(
            num_envs,
            dtype=torch.int32,
            device=device,
        )
        env._forbidden_contact_cooldown = torch.zeros(
            num_envs,
            device=device,
        )
        env._forbidden_contact_episode_time = torch.zeros(
            num_envs,
            device=device,
        )

    counter = env._forbidden_contact_count
    cooldown = env._forbidden_contact_cooldown
    episode_time = env._forbidden_contact_episode_time

    episode_steps = getattr(env, "episode_length_buf", None)
    if episode_steps is not None:
        new_episode = episode_steps <= 1

        counter[new_episode] = 0
        cooldown[new_episode] = 0.0
        episode_time[new_episode] = 0.0

    episode_time.add_(dt)
    cooldown.sub_(dt).clamp_(min=0.0)

    count_now = (
        forbidden_contact
        & (cooldown <= 0.0)
        & (episode_time >= grace_s)
    )

    # Первый контакт сразу даёт 1. Продолжающийся контакт спустя
    # contact_interval_s даст 2 и вызовет termination.
    counter.add_(count_now.to(torch.int32))
    cooldown[count_now] = contact_interval_s

    return counter > maximum_contacts
