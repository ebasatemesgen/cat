# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class Solo12FlatRslRlPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # runner
    seed = 0
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 250
    experiment_name = "solo_cat"
    run_name = ""
    logger = "tensorboard"
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
    clip_actions = None

    # CaT-specific runner switch consumed by scripts/rsl_rl/train.py.
    use_cat = True

    # Explicitly map actor/critic observation groups.
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }

    # policy
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # algorithm
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=2.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=6,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
