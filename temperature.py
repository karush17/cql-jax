"""Implements the temperature learning module."""

from typing import Tuple

import functools
import jax
import jax.numpy as jnp

from flax import linen as nn
from common import InfoDict, Model


class Temperature(nn.Module):
    """Class for tuning the temperature variable.
    
    Attributes:
        initial_temperature: initial temperature value.
    """
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        """Executes the forward pass."""
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)


def update(temp: Model, entropy: float,
           target_entropy: float) -> Tuple[Model, InfoDict]:
    """Updates the temperature value."""
    def temperature_loss_fn(temp_params):
        temperature = temp.apply_fn({'params': temp_params})
        temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}

    new_temp, info = temp.apply_gradient(temperature_loss_fn)

    return new_temp, info


class Lagrange(nn.Module):
    """Class for tuning the lagrange multiplier.
    
    Attributes:
        initial_lagrange: initial value of the lagrange variable.
    """
    initial_lagrange: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        """Executes the forward pass."""
        log_lag = self.param('log_lag',
                              init_fn=lambda key: jnp.full(
                                  (), jnp.log(self.initial_lagrange)))
        return jnp.exp(log_lag)


@functools.partial(jax.jit, static_argnames=('update_target'))
def update_lag(lag: Model, gap1: float, gap2: float,
           target_gap: float) -> Tuple[Model, InfoDict]:
    """Updates the lagrange variable."""
    def lagrange_loss_fn(lag_params):
        lagrange = lag.apply_fn({'params': lag_params})
        lag1_loss = lagrange * (gap1 - target_gap)
        lag2_loss = lagrange * (gap2 - target_gap)
        lag_loss = (lag1_loss + lag2_loss)*0.5
        return lag_loss, {'lagrange': lagrange, 'lag_loss': lag_loss}

    new_lag, info = lag.apply_gradient(lagrange_loss_fn)

    return new_lag, info
