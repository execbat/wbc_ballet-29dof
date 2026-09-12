"""RSL-RL PPO configuration owned by the flip task."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from .flip_curriculum_cfg import PPO_STEPS_PER_ITERATION


def flip_ppo_runner_cfg():
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True, distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1., "std_type": "scalar"}),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1., use_clipped_value_loss=True, clip_param=.2, entropy_coef=.01, num_learning_epochs=5, num_mini_batches=4, learning_rate=1.e-3, schedule="adaptive", gamma=.99, lam=.95, desired_kl=.01, max_grad_norm=1.),
        experiment_name="g1_29dof_flip", run_name="flip", save_interval=250,
        num_steps_per_env=PPO_STEPS_PER_ITERATION, max_iterations=200_000,
    )
