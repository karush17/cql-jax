"""Implements the training and loss function for critic."""

from typing import Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.lax as lax

from jax.scipy.special import logsumexp
from jax.random import uniform
from flax import linen as nn
from utils import Batch
from common import MLP, InfoDict, Model, Params, PRNGKey

NUM_REPEAT = 5

def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    """Implements the target update."""
    new_target_params = jax.tree_multimap(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)
    return target_critic.replace(params=new_target_params)

def get_values(critic: Model, state: jnp.ndarray, actions: jnp.ndarray,
               critic_params: Params):
    """Fetches the value from value network."""
    state_shape = state.shape[0]
    state_temp = jnp.repeat(jnp.expand_dims(state, axis=1), NUM_REPEAT, axis=1)
    state_temp = jnp.reshape(state_temp, (state_shape*NUM_REPEAT, state.shape[1]))
    preds1, preds2 = critic.apply_fn({'params': critic_params}, state_temp, actions)
    preds1 = jnp.reshape(preds1, (state_shape, NUM_REPEAT, 1))
    preds2 = jnp.reshape(preds2, (state_shape, NUM_REPEAT, 1))
    return preds1, preds2

def get_actions(actor: Model, state: jnp.ndarray):
    """Fetches the actions from the actor."""
    state_shape = state.shape[0]
    state_temp = jnp.repeat(jnp.expand_dims(state, axis=1), NUM_REPEAT, axis=1)
    state_temp = jnp.reshape(state_temp, (state_shape*NUM_REPEAT, state.shape[1]))
    actions = actor(state_temp)
    return actions

def update(key: PRNGKey, actor: Model, critic: Model, target_critic: Model,
           temp: Model, batch: Batch, discount: float,
           soft_critic: bool) -> Tuple[Model, InfoDict]:
    """Updates the value network with Bellman loss."""
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    next_q1, next_q2 = target_critic(batch.next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)

    target_q = batch.rewards + discount * batch.masks * next_q

    if soft_critic:
        target_q -= discount * batch.masks * temp() * next_log_probs

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply_fn({'params': critic_params}, batch.observations,
                                 batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info


def update_cql(key: PRNGKey, actor: Model, critic: Model, target_critic: Model,
           temp: Model, lag: jnp.ndarray, lagrange_thresh: float,
           min_q_weight: float, batch: Batch, discount: float,
           soft_critic: bool) -> Tuple[Model, InfoDict]:
    """Updates the value network with CQL loss."""
    dist = get_actions(actor, batch.observations)
    actions = dist.sample(seed=key)
    log_probs = jnp.reshape(dist.log_prob(actions),
                            (batch.observations.shape[0], NUM_REPEAT, 1))

    next_dist = get_actions(actor, batch.next_observations)
    next_actions = next_dist.sample(seed=key)
    next_log_probs = jnp.reshape(dist.log_prob(next_actions),
                                 (batch.observations.shape[0], NUM_REPEAT, 1))

    rand_actions = uniform(key, (actions.shape[0],actions.shape[1]),
                           minval=-1, maxval=1)
    rand_density = jnp.log(0.5**actions.shape[-1])

    next_dist = actor(batch.next_observations)
    actions_next = next_dist.sample(seed=key)
    log_probs_next = next_dist.log_prob(actions_next)
    q1_next, q2_next = target_critic(batch.next_observations, actions_next)
    next_q = jnp.minimum(q1_next, q2_next)
    target_q = batch.rewards + discount * batch.masks * next_q
    if soft_critic:
        target_q -= discount * batch.masks * temp() * log_probs_next

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = get_values(critic, batch.observations, actions, critic_params)
        next_q1, next_q2 = get_values(target_critic, batch.next_observations,
                                                     next_actions, critic_params)
        rand_q1, rand_q2 = get_values(critic, batch.observations,
                                      rand_actions, critic_params)

        cat_q1 = jnp.concatenate([rand_q1 - rand_density,
                        next_q1 - next_log_probs, q1 - log_probs], axis=1)
        cat_q2 = jnp.concatenate([rand_q2 - rand_density,
                        next_q2 - next_log_probs, q2 - log_probs], axis=1)

        gap1 = logsumexp(cat_q1, axis=1).mean() * min_q_weight
        gap2 = logsumexp(cat_q2, axis=1).mean() * min_q_weight

        lag_val = lax.clamp(0.0, lag, 1e6)
        gap1 = lag_val * (gap1 - lagrange_thresh)
        gap2 = lag_val * (gap2 - lagrange_thresh)

        q1_pred, q2_pred = critic.apply_fn({'params': critic_params},
                                           batch.observations, batch.actions)
        q1_loss = gap1 - q1_pred.mean() * min_q_weight
        q2_loss = gap2 - q2_pred.mean() * min_q_weight

        q1_loss = q1_loss + ((q1_pred - target_q)**2).mean()
        q2_loss = q2_loss + ((q2_pred - target_q)**2).mean()

        critic_loss = 0.5*(q1_loss + q2_loss)
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1_pred.mean(),
            'q2': q2_pred.mean(),
            'gap1': gap1,
            'gap2': gap2
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info


class ValueCritic(nn.Module):
    """Implements the value network architecture.
    
    Attributes:
        hidden_dims: number of hidden units.
    """
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        """Executes the forward pass."""
        critic = MLP((*self.hidden_dims, 1))(observations)
        return jnp.squeeze(critic, -1)


class Critic(nn.Module):
    """Implements the critic network module.
    
    Attributes:
        hidden_dims: number of hidden dimensions.
        activations: activation function.
    """
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        """Executes the forward pass."""
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1),
                     activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    """Implements the double critic network.
    
    Attributes:
        hidden_dims: number of hidden dimensions.
        activations: activation function.
    """
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Executes the forward pass."""
        critic1 = Critic(self.hidden_dims,
                         activations=self.activations)(observations, actions)
        critic2 = Critic(self.hidden_dims,
                         activations=self.activations)(observations, actions)
        return critic1, critic2
