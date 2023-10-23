"""Implements the configurations for CQL algorithm."""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
    """Stores the agent hyperparameters."""
    config = ml_collections.ConfigDict()

    config.actor_lr = 1e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.tau = 0.005
    config.target_update_period = 1

    config.init_temperature = 1.0
    config.target_entropy = None
    config.lagrange_thresh = 0.0
    config.init_lagrange = 1.0
    config.start_step = 40000

    return config
