"""Implements the common training modules."""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import os
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax


PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any
InfoDict = Dict[str, float]

def default_init(scale: Optional[float] = jnp.sqrt(2)) -> jnp.ndarray:
    """Implements the parameter initialization."""
    return nn.initializers.orthogonal(scale)


class MLP(nn.Module):
    """Implements the MLP architecture.
    
    Attributes:
        hidden_dims: number of hidden dimensions.
        activations: activation function.
        activate_final: whether to use activation after final layer.
        dropout_rate: probability of dropout.
    """
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Executes the forward pass."""
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x


@flax.struct.dataclass
class Model:
    """Implements the train state of the model.
    
    Attributes:
        step: training step.
        apply_fn: function for updating networks.
        params: trainable parameters.
        tx: gradient transformation.
        opt_state: state of the optimizer.
    """
    step: int
    apply_fn: Callable[..., Any] = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None) -> 'Model':
        """Creates the object."""
        variables = model_def.init(*inputs)

        _, params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def.apply,
                   params=params,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        """Executes the forward pass."""
        return self.apply_fn({'params': self.params}, *args, **kwargs)

    def apply_gradient(
            self,
            loss_fn: Callable[[Params], Any],
            has_aux: bool = True) -> Union[Tuple['Model', Any], 'Model']:
        """Applies the gradient transformation."""
        grad_fn = jax.grad(loss_fn, has_aux=has_aux)
        if has_aux:
            grads, aux = grad_fn(self.params)
        else:
            grads = grad_fn(self.params)

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        new_model = self.replace(step=self.step + 1,
                                 params=new_params,
                                 opt_state=new_opt_state)
        if has_aux:
            return new_model, aux
        else:
            return new_model

    def save(self, save_path: str):
        """Saves the model."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'Model':
        """Loads the model."""
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)
