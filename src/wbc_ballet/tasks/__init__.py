from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .ballet import ballet_ppo_runner_cfg, make_ballet_env_cfg

register_mjlab_task(
    task_id="Mjlab-Ballet-Flat-Unitree-G1-29DoF",
    env_cfg=make_ballet_env_cfg(),
    play_env_cfg=make_ballet_env_cfg(play=True),
    rl_cfg=ballet_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Ballet-Rough-Unitree-G1-29DoF",
    env_cfg=make_ballet_env_cfg(rough=True),
    play_env_cfg=make_ballet_env_cfg(play=True, rough=True),
    rl_cfg=ballet_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
