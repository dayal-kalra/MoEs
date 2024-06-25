#Some imports
import jax
import jax.numpy as jnp
import optax
from typing import Any, Callable, Sequence, Tuple
from functools import partial
from flax import core
from flax import struct
from jax.numpy.linalg import norm
from jax.experimental import sparse
import numpy as np

class TrainState(struct.PyTreeNode):
    step: int
    apply_fn: Callable = struct.field(pytree_node = False)
    params: core.FrozenDict[str, Any]
    opt: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value."""
        updates, new_opt_state = self.opt.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step = self.step + 1, params = new_params,  opt_state = new_opt_state, **kwargs,)

    def update_learning_rate(self, *, learning_rate):
        """ Updates the learning rate"""
        self.opt_state.hyperparams['learning_rate'] = learning_rate
        return

    def get_optimizer_hparams(self,):
        return self.opt_state.hyperparams

    @classmethod
    def create(cls, *, apply_fn, params, opt, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = opt.init(params)
        return cls(step = 0, apply_fn = apply_fn, params = params, opt = opt, opt_state = opt_state, **kwargs, )
        
def compute_accuracy(logits, labels, mask):
    """ Accuracy, used while measuring the state"""
    # Get the label of the one-hot encoded target
    logits_flat = logits.reshape(-1, logits.shape[-1]) # shape (B*T, C)
    labels_flat = labels.reshape(-1) # shape (B*T,)
    labels_flat = jax.nn.one_hot(labels_flat, logits.shape[-1]) # (B*T, C)
    mask_flat = mask.reshape(-1) # shape (B*T, )
    # get the argmax along axis -1
    labels_class = jnp.argmax(labels_flat, axis = -1)
    # get the argmax along axis -1
    predicted_class = jnp.argmax(logits_flat, axis = -1)
    masked_accuracy = jnp.where(mask_flat, (predicted_class == labels_class), 0)
    total_accuracy = jnp.sum(masked_accuracy)
    num_masked_elements = jnp.sum(mask_flat)
    return total_accuracy / num_masked_elements

@partial(jax.jit, static_argnums = 3)
def compute_metrics(state, params, batch, loss_function):
    # labels is (B, T)
    inputs, labels, mask = batch
    # forward pass
    logits = state.apply_fn({'params': params}, inputs)
    # compute loss
    loss = loss_function(logits = logits, labels = labels, mask = mask)
    # compute accuracy
    accuracy = compute_accuracy(logits, labels, mask)
    return loss, accuracy

@partial(jax.jit, static_argnums = 3)
def compute_loss(state, params, batch, loss_function):
    # labels is (B, T)
    inputs, labels, mask = batch
    # forward pass
    logits = state.apply_fn({'params': params}, inputs)
    # compute loss
    loss = loss_function(logits = logits, labels = labels, mask = mask)
    return loss

@partial(jax.jit, static_argnums = 2)
def grads_step(state: TrainState, batch: Tuple, loss_function):
    "Estimates the gradients for a given batch"
    # decouple inputs and outputs and reshape
    inputs, labels, mask = batch
    
    # loss function 
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, inputs)
        loss = loss_function(logits = logits, labels = labels, mask = mask)
        return loss, logits

    #calculate the gradients and loss
    grad_fn = jax.value_and_grad(loss_fn, has_aux = True)

    (loss, logits), grads = grad_fn(state.params)
    return grads, loss

@partial(jax.jit, static_argnums = 3)
def loss_step(state: TrainState, batch: Tuple, params, loss_function):
    "Compute loss for a single batch"
    inputs, labels, mask = batch
    logits = state.apply_fn({'params': params}, inputs)
    loss = loss_function(logits = logits, labels = labels, mask = mask)
    return logits, loss

@partial(jax.jit, static_argnums = 2)
def train_step(state: TrainState, batch: Tuple, loss_function):
    "Train the model with a given batch"
    # decouple inputs and outputs and reshape
    inputs, labels, mask = batch
    
    # loss function 
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, inputs)
        loss = loss_function(logits = logits, labels = labels, mask = mask)
        return loss, logits

    #calculate the gradients and loss
    grad_fn = jax.value_and_grad(loss_fn, has_aux = True)

    (loss, logits), grads = grad_fn(state.params)
    #update the state
    state = state.apply_gradients(grads = grads)
    return state, grads, logits, loss


@partial(jax.jit, static_argnums = 2)
def train_sharpness_lobpcg_step(state: TrainState, batch: Tuple, loss_function, vs, m_iter: int = 1000, tol = 1e-09):
    "Train the model for a single batch and estimate sharpness"

    # decouple inputs and outputs and reshape
    inputs, labels, mask = batch

    # loss function 
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, inputs)
        loss = loss_function(logits = logits, labels = labels, mask = mask)
        return loss, logits
    
    flat_params, rebuild_fn = jax.flatten_util.ravel_pytree(state.params)

    def loss_fn_flat(flat_params):
        unflat_params = rebuild_fn(flat_params)
        loss, _ = loss_fn(unflat_params)
        return loss

    def hvp(flat_params, v):
        return jax.jvp(jax.grad(loss_fn_flat), [flat_params], [v])[1]

    body_hvp = jax.tree_util.Partial(hvp, flat_params)
    body_hvp = jax.vmap(body_hvp, 1, -1)

    vs = vs / norm(vs, axis = -1, keepdims = True)
    eigs, eigvs, n_iter = sparse.linalg.lobpcg_standard(body_hvp, vs, m = m_iter, tol = tol)

    #calculate the gradients and loss
    grad_fn = jax.value_and_grad(loss_fn, has_aux = True)
    (loss, logits), grads = grad_fn(state.params)
    #update the state
    state = state.apply_gradients(grads = grads)
    return state, logits, grads, loss, eigs, eigvs, n_iter


@partial(jax.jit, static_argnums = 2)
def hessian_lobpcg_step(state: TrainState, batch: Tuple, loss_function, vs, m_iter: int = 1000, tol = 1e-09):
    "Train the model for a single batch and estimate sharpness"

    # decouple inputs and outputs and reshape
    inputs, labels, mask = batch

    # loss function 
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, inputs)
        loss = loss_function(logits = logits, labels = labels, mask = mask)
        return loss, logits
    
    flat_params, rebuild_fn = jax.flatten_util.ravel_pytree(state.params)

    def loss_fn_flat(flat_params):
        unflat_params = rebuild_fn(flat_params)
        loss, _ = loss_fn(unflat_params)
        return loss

    def hvp(flat_params, v):
        return jax.jvp(jax.grad(loss_fn_flat), [flat_params], [v])[1]

    body_hvp = jax.tree_util.Partial(hvp, flat_params)
    body_hvp = jax.vmap(body_hvp, 1, -1)

    vs = vs / norm(vs, axis = -1, keepdims = True)
    eigs, eigvs, n_iter = sparse.linalg.lobpcg_standard(body_hvp, vs, m = m_iter, tol = tol)

    return eigs, eigvs, n_iter


def data_stream(seed, tokens_lst, batch_size):
    """ Creates a data stream with a predifined batch size. No augmentation for LM."""
    num_examples = len(tokens_lst)
    num_batches = estimate_num_batches(num_examples, batch_size)
    rng = np.random.RandomState(seed)

    while True:
        perm = rng.permutation(num_examples)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size: (i + 1)*batch_size]
            x_batch = tokens_lst[batch_idx]
            yield x_batch

def estimate_num_batches(num_train, batch_size):
    "Estimates number of batches using dataset and batch size"
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    return num_batches

def generate(state, sample_key, start_tokens, max_length, END_TOKEN, temperature = 1.0, top_k = None):
    """
    Generates text from a starting sequence of tokens using a trained model.

    Args:
        model: The trained Flax model.
        params: The parameters of the model.
        start_tokens: Array of token IDs to start generation, shape (T, ).
        max_length: Maximum length of the generated sequence.
        temperature: Sampling temperature, controls randomness.
        top_k: Limits sampling to the top k probabilities, enhancing diversity.

    Returns:
        An array of token IDs representing the generated text.
    """
    # Convert start tokens to a JAX array if necessary
    tokens = [t for t in start_tokens]

    @jax.jit
    def step(tokens, key):
        # Model prediction
        logits = state.apply_fn({'params': state.params}, tokens.reshape(1, -1))
        logits = logits[0, -1, :] / temperature  # Get the logits for the last token in the sequence
        # Apply top-k filtering if specified
        # if top_k is not None:
        #     top_values, top_indices = jax.lax.top_k(logits, top_k)
        #     logits = jax.nn.one_hot(top_indices, logits.size) @ top_values
        next_token = jax.random.categorical(key, logits) 
        return next_token

    # Initialize the sequence and run the sampling loop
    for _ in range(max_length - len(start_tokens)):
        key, sample_key = jax.random.split(sample_key, 2)
        next_token = step(jnp.asarray(tokens, dtype = jnp.int32), key)
        tokens.append(next_token.item())
        if tokens[-1] == END_TOKEN:
            break
    return jnp.asarray(tokens, dtype = jnp.int32)
