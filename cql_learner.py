"""Implements the CQL algorithm."""

from typing import Optional, Sequence, Tuple

import functools
import jax
import jax.numpy as jnp
import numpy as np
import optax

import temperature

from actor import NormalTanhPolicy, sample_actions, update_bc
from actor import update as update_actor
from critic import DoubleCritic, target_update
from critic import update_cql as update_cql_critic
from utils import Batch
from common import InfoDict, Model, PRNGKey


@functools.partial(jax.jit, static_argnames=('update_target'))
def _update_jit(
    rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
    temp: Model, lag: jnp.ndarray, batch: Batch, discount: float, tau: float,
    target_entropy: float, lagrange_thresh: float, min_q_weight: float,
    update_target: bool
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:
    """Updates the policy and value networks."""

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_cql_critic(key, actor, critic,
                                                target_critic, temp, lag,
                                                lagrange_thresh, min_q_weight,
                                                batch, discount, soft_critic=True)

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, critic, temp, batch)
    new_temp, alpha_info = temperature.update(temp, actor_info['entropy'],
                                              target_entropy)

    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }


@functools.partial(jax.jit, static_argnames=('update_target'))
def _bc_jit(
    rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
    temp: Model, lag: jnp.ndarray, batch: Batch, discount: float, tau: float,
    target_entropy: float, lagrange_thresh: float, min_q_weight: float,
    update_target: bool
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:
    """Updates the policy network using BC."""

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_cql_critic(key, actor, critic,
                                                target_critic, temp, lag,
                                                lagrange_thresh, min_q_weight,
                                                batch, discount, soft_critic=True)

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_bc(key, actor, critic, temp, batch)
    new_temp, alpha_info = temperature.update(temp, actor_info['entropy'],
                                              target_entropy)

    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }

class CQLLearner(object):
    """Implementation of CQL.
    
    Attributes:
        target_entropy: true entropy value for SAC.
        tau: soft update constant.
        target_update_period: intervals till target update.
        discount: discount factor.
        min_q_weight: weight of Q value regularizer.
        lagrange_thresh: lagrange threshold.
        with_lagrange: whether to train with lagrange tuning.
        start_step: training start iteration.
        actor: policy network.
        critic: value network.
        target_critic: target value network.
        rng: jax random key.
        step: current training step.
        lag_val: lagrange value.
    """
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-4,
                 temp_lr: float = 1e-4,
                 min_q_weight: float = 10,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.01,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 start_step: int = 5000,
                 lagrange_thresh: float = -1,
                 init_lagrange: float = 1.0,
                 init_temperature: float = 1.0):
        """Initializes the class object."""

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount
        self.min_q_weight = min_q_weight
        self.lagrange_thresh = lagrange_thresh
        self.with_lagrange = lagrange_thresh > 0.0
        self.start_step = start_step

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key, lag_key = jax.random.split(rng, 5)
        actor_def = NormalTanhPolicy(hidden_dims, action_dim)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))

        critic_def = DoubleCritic(hidden_dims)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        temp = Model.create(temperature.Temperature(init_temperature),
                            inputs=[temp_key],
                            tx=optax.adam(learning_rate=temp_lr))

        self.lag = None
        if self.with_lagrange:
            lag = Model.create(temperature.Lagrange(init_lagrange),
                                inputs=[lag_key],
                                tx=optax.adam(learning_rate=temp_lr))
            self.lag = lag

        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.lag_val = init_lagrange
        self.rng = rng

        self.step = 1

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        """Samples an action from the agent."""
        rng, actions = sample_actions(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        """Updates the actor and critic."""
        self.step += 1

        if self.step > self.start_step:
            new_rng, new_actor, new_critic, \
                new_target_critic, new_temp, info = _update_jit(
                self.rng, self.actor, self.critic, \
                    self.target_critic, self.temp,
                self.lag_val, batch, self.discount, \
                    self.tau, self.target_entropy,
                self.lagrange_thresh, self.min_q_weight, \
                    self.step % self.target_update_period == 0)
        else:
            new_rng, new_actor, new_critic, \
                new_target_critic, new_temp, info = _bc_jit(
                self.rng, self.actor, self.critic, \
                    self.target_critic, self.temp,
                self.lag_val, batch, self.discount, \
                    self.tau, self.target_entropy,
                self.lagrange_thresh, self.min_q_weight, \
                    self.step % self.target_update_period == 0)


        if self.with_lagrange:
            new_lag, _ = temperature.update_lag(self.lag, info['gap1'],
                                                info['gap2'],
                                                self.lagrange_thresh)
            self.lag = new_lag
            self.lag_val = self.lag()

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp

        return info
