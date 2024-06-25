from typing import NamedTuple
import jax.flatten_util
import jax.numpy as jnp
import optax
import jax
import flax


######## custom Adam optimizer ##########

class AdamState(NamedTuple):
    mu: optax.Updates  # Moving average of the gradients
    nu: optax.Updates  # Moving average of the squared gradients
    count: jnp.ndarray  # Timestep


def custom_adamw(learning_rate: float = 0.001, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.0001, grads = None):

    def init_fn(params):
        mu = jax.tree_map(jnp.zeros_like, params)
        nu = jax.tree_map(jnp.zeros_like, params)
        if grads is not None:
            nu = jax.tree_map(lambda x: x**2, grads)  
        return AdamState(mu = mu, nu = nu, count = jnp.zeros([]))

    def update_fn(grads, state, params, learning_rate = learning_rate, b1 = b1, b2 = b2, eps = eps, weight_decay = weight_decay):
        mu, nu, count = state.mu, state.nu, state.count + 1
        # flatten everything
        flat_grads, rebuild_fn = jax.flatten_util.ravel_pytree(grads)
        flat_params, _ = jax.flatten_util.ravel_pytree(params)  # used for weight decay
        flat_mu, rebuild_fn = jax.flatten_util.ravel_pytree(mu)
        flat_nu, rebuild_fn = jax.flatten_util.ravel_pytree(nu)

        flat_mu_next = b1 * flat_mu + (1 - b1) * flat_grads
        flat_mu_hat = flat_mu_next / (1 - b1 ** count)

        flat_nu_next = b2 * flat_nu + (1 - b2) * (flat_grads**2)
        flat_nu_hat = flat_nu_next / (1 - b2 ** count)

        flat_updates =  -learning_rate * (flat_mu_hat / (jnp.sqrt(flat_nu_hat) + eps) + weight_decay * flat_params )
            
        updates = rebuild_fn(flat_updates)
        mu_next = rebuild_fn(flat_mu_next)
        nu_next = rebuild_fn(flat_nu_next)

        return updates, AdamState(mu = mu_next, nu = nu_next, count = count)

    return optax.GradientTransformation(init_fn, update_fn)


def sparse_adamw(learning_rate: float = 0.001, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.0001, update_freq = 1, grads = None):

    def init_fn(params):
        mu = jax.tree_map(jnp.zeros_like, params)
        nu = jax.tree_map(jnp.zeros_like, params)
        if grads is not None:
            nu = jax.tree_map(lambda x: x**2, grads)  
        return AdamState(mu = mu, nu = nu, count = jnp.zeros([]))

    def update_fn(grads, state, params, learning_rate = learning_rate, b1 = b1, b2 = b2, eps = eps, weight_decay = weight_decay, update_freq = update_freq):
        mu, nu, count = state.mu, state.nu, state.count + 1
        # flatten everything
        flat_grads, rebuild_fn = jax.flatten_util.ravel_pytree(grads)
        flat_params, _ = jax.flatten_util.ravel_pytree(params)  # used for weight decay
        flat_mu, rebuild_fn = jax.flatten_util.ravel_pytree(mu)
        flat_nu, rebuild_fn = jax.flatten_util.ravel_pytree(nu)

        flat_mu_next = b1 * flat_mu + (1 - b1) * flat_grads
        flat_mu_hat = flat_mu_next / (1 - b1 ** count)

        def update_nu():
            flat_nu_next = b2 * flat_nu + (1 - b2) * (flat_grads**2)
            flat_nu_hat = flat_nu_next / (1 - b2 ** count)
            return flat_nu_next, flat_nu_hat

        def maintain_nu():
            return flat_nu, flat_nu

        condition = jnp.logical_or((count + 1) % update_freq == 0, count < 100)
        flat_nu_next, flat_nu_hat = jax.lax.cond(condition, update_nu, maintain_nu)
        # update the model
        flat_updates =  -learning_rate * (flat_mu_hat / (jnp.sqrt(flat_nu_hat) + eps) + weight_decay * flat_params )
            
        updates = rebuild_fn(flat_updates)
        mu_next = rebuild_fn(flat_mu_next)
        nu_next = rebuild_fn(flat_nu_next)

        return updates, AdamState(mu = mu_next, nu = nu_next, count = count)

    return optax.GradientTransformation(init_fn, update_fn)




def custom_adam_lr(lr_pytree, learning_rate: float = 0.001, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8, grads = None):

    def init_fn(params):        
        mu = jax.tree_map(jnp.zeros_like, params)
        nu = jax.tree_map(jnp.zeros_like, params)
        if grads is not None:
            nu = jax.tree_map(lambda x: x**2, grads) # 
        return AdamState(mu = mu, nu = nu, count = jnp.zeros([]))

    def update_fn(grads, state, params = None, learning_rate = learning_rate, b1 = b1, b2 = b2, eps = eps):
        
        mu, nu, count = state.mu, state.nu, state.count + 1
        # flatten grads, mu, nu
        flat_grads, rebuild_fn = jax.flatten_util.ravel_pytree(grads)
        flat_mu, rebuild_fn = jax.flatten_util.ravel_pytree(mu)
        flat_nu, rebuild_fn = jax.flatten_util.ravel_pytree(nu)

        # update mu
        flat_mu_next = b1 * flat_mu + (1 - b1) * flat_grads
        flat_mu_hat = flat_mu_next / (1 - b1 ** count)
        
        # update new
        flat_nu_next = b2 * flat_nu + (1 - b2) * (flat_grads**2)
        flat_nu_hat = flat_nu_next / (1 - b2 ** count)

        # updates except the learning rate
        flat_updates =  -flat_mu_hat / (jnp.sqrt(flat_nu_hat) + eps)
        
        # unflatten back
        updates = rebuild_fn(flat_updates)
        mu_next = rebuild_fn(flat_mu_next)
        nu_next = rebuild_fn(flat_nu_next)

        # estimate the learning rate dictionary

        # multiply with learning rate
        updates = jax.tree_map(lambda lr,g: learning_rate*lr*g, lr_pytree, updates)

        return updates, AdamState(mu = mu_next, nu = nu_next, count = count)

    return optax.GradientTransformation(init_fn, update_fn)


def flatten_pytree(pytree, prefix = ''):
    flat_dict = {}
    for key, value in pytree.items():
        # Construct the new key path
        new_key = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            # If the value is a dictionary, recurse further
            flat_dict.update(flatten_pytree(value, new_key))
        else:
            # Otherwise, store the value with its accumulated key path
            flat_dict[new_key] = value
    return flat_dict


######## custom SGD-M optimizer ##########

class SgdState(NamedTuple):
    mu: optax.Updates  # moving average of gradients
    count: jnp.ndarray  # Timestep

def custom_sgdm(learning_rate: float = 0.01, momentum: float = 0.0):

    def init_fn(params):
        mu = jax.tree_map(jnp.zeros_like, params)
        return SgdState(mu = mu, count = jnp.zeros([]))

    def update_fn(grads, state, params = None, learning_rate = learning_rate, momentum = momentum):
        mu, count = state.mu, state.count + 1

        # flatten everything
        flat_grads, rebuild_fn = jax.flatten_util.ravel_pytree(grads)
        flat_mu, _ = jax.flatten_util.ravel_pytree(mu)

        # Update momentum
        flat_mu_next = momentum * flat_mu + flat_grads
        # Compute parameter updates
        flat_updates = -learning_rate * flat_mu_next

        updates = rebuild_fn(flat_updates)
        mu_next = rebuild_fn(flat_mu_next)

        return updates, SgdState(mu = mu_next, count = count)

    return optax.GradientTransformation(init_fn, update_fn)


def custom_sgdm_lr(lr_pytree, learning_rate: float = 1.0, momentum: float = 0.0):

    def init_fn(params):
        mu = jax.tree_map(jnp.zeros_like, params)
        return SgdState(mu = mu, count = jnp.zeros([]))

    def update_fn(grads, state, params = None, learning_rate = learning_rate, momentum = momentum):
        mu, count = state.mu, state.count + 1

        # flatten everything
        flat_grads, rebuild_fn = jax.flatten_util.ravel_pytree(grads)
        flat_mu, _ = jax.flatten_util.ravel_pytree(mu)

        # Update momentum
        flat_mu_next = momentum * flat_mu + flat_grads
        # Compute parameter updates
        flat_updates = -learning_rate * flat_mu_next

        updates = rebuild_fn(flat_updates)
        mu_next = rebuild_fn(flat_mu_next)

        # multiply with learning rate
        updates = jax.tree_map(lambda lr,g: lr*g, lr_pytree, updates)

        return updates, SgdState(mu = mu_next, count = count)

    return optax.GradientTransformation(init_fn, update_fn)
