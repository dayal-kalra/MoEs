import optax
import jax
import jax.numpy as jnp

def mse_loss(logits, labels):
    """ MSE loss used while measuring the state"""
    return 0.5 * jnp.mean((logits - labels) ** 2)

def cross_entropy_loss(logits, labels):
  return optax.softmax_cross_entropy(logits = logits, labels = labels).mean()

def cross_entropy_loss_integer_labels(logits, labels):
  return optax.softmax_cross_entropy_with_integer_labels(logits = logits, labels = labels).mean()


def masked_cross_entropy_loss(logits, labels, mask):
    """
    Computes cross-entropy loss for only the masked positions using optax's softmax_cross_entropy.

    Args:
        logits: Logits from the model, shape (B, T, C).
        labels: Ground truth labels, shape (B, T).
        mask: A boolean array of the same shape as labels, where True indicates that
              the loss should be calculated for that position.

    Returns:
        Average loss for the masked positions.
    """
    # Calculate the softmax cross-entropy loss
    logits_flat = logits.reshape(-1, logits.shape[-1]) # shape (B*T, C)
    labels_flat = labels.reshape(-1) # shape (B*T,)
    mask_flat = mask.reshape(-1) # # shape (B*T, )
    
    loss = optax.softmax_cross_entropy(logits_flat, jax.nn.one_hot(labels_flat, logits.shape[-1]))
    
    # Mask the loss
    masked_loss = jnp.where(mask_flat, loss, 0)

    # Compute the average loss only over the masked elements
    total_loss = jnp.sum(masked_loss)
    num_masked_elements = jnp.sum(mask_flat)
    
    return total_loss / num_masked_elements

