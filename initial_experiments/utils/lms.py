from typing import Any, Tuple, Type

import flax.linen as nn
import jax.numpy as jnp
from flax import struct
from jax import Array, tree_leaves
from functools import partial
import jax

def count_parameters(params):
    "counts the number of parameters of a model"
    return sum(x.size for x in tree_leaves(params))

class SimpleDense(nn.Module):
    """ muP Readout layer """
    fan_out: int 
    use_bias: bool = True # bool for bias
    varw: float = 2.0 # variance
    dtype: Type = jnp.float32

    @nn.compact
    def __call__(self, x):
        # 1 / fan_in init
        kernel_init = nn.initializers.variance_scaling(scale = self.varw, distribution = 'truncated_normal', mode = 'fan_in')
        x = nn.Dense(self.fan_out, kernel_init = kernel_init, use_bias = self.use_bias, dtype = self.dtype)(x)
        return x

class muDense(nn.Module):
    """ muP Readout layer """
    fan_out: int # num_classes
    use_bias: bool = True # bool for bias
    varw: float = 2.0 # variance
    dtype: Type = jnp.float32

    @nn.compact
    def __call__(self, x):
        fan_in = x.shape[-1]
        # 1 / fan_out init
        kernel_init = nn.initializers.variance_scaling(scale = self.varw, distribution = 'truncated_normal', mode = 'fan_out')
        x = nn.Dense(self.fan_out, kernel_init = kernel_init, use_bias = self.use_bias, dtype = self.dtype)(x)
        # sqrt(fan_out / fan_in) multiplier
        x *= jnp.sqrt(self.fan_out / fan_in)
        return x
    
class muReadout(nn.Module):
    """ muP Readout layer """
    fan_out: int # num_classes
    use_bias: bool = True
    varw: float = 1.0
    dtype: Type = jnp.float32

    @nn.compact
    def __call__(self, x):
        fan_in = x.shape[-1]
        # 1 / fan_in initialization
        kernel_init = nn.initializers.variance_scaling(scale = 1.0, distribution = 'truncated_normal', mode = 'fan_in')
        x = nn.Dense(self.fan_out, kernel_init = kernel_init, use_bias = self.use_bias)(x)
        #  sqrt(1 / fan_in) multiplier
        x *= jnp.sqrt(1 / fan_in)
        return x

# struct. dataclass ensures that once this config is created it cannot be changed (i.e., it is immutable)
@struct.dataclass
class GPTConfig:
    vocab_size: int # vocabulary size
    cntxt_len: int # the block size / context length (T)
    n_blocks: int # number of layers (L)
    n_head: int # number of heads (n_head)
    n_embd: int # embedding dimension (n)
    ffwd_upscale: int = 4  # upscaling factor for MLPs
    varw_scale: float = 1.0
    resd_scale: float = 0.0 # n_layers**resd_factor
    readout_scale: float = 0.0 # n_embd**readout_scale
    attn_scale: float = 0.5 # attention scale
    use_bias: bool = True # wheather to use bias or not
    dtype: Any = jnp.float32 # datatype of the model
    # MoE stuff
    num_experts: int = 4
    top_k: int = 3
    norm_topk_prob: bool = False

class SinusoidalPositionEmbed(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, length: int) -> Array:
        position = jnp.arange(length)[:, None]
        div_term = jnp.exp(jnp.arange(0, self.config.n_embd, 2) * -(jnp.log(10000.0) / self.config.n_embd))
        pos_encoding = jnp.zeros((length, self.config.n_embd))
        pos_encoding = pos_encoding.at[:, 0::2].set(jnp.sin(position * div_term))
        pos_encoding = pos_encoding.at[:, 1::2].set(jnp.cos(position * div_term))
        return pos_encoding

variance_scaling = partial(nn.initializers.variance_scaling, mode = 'fan_in', distribution = 'truncated_normal')

""" MLP block """
class MLP_Block(nn.Module):
    config: GPTConfig
    Dense: nn.Module

    def setup(self):
        
        self.fc = self.Dense(self.config.ffwd_upscale * self.config.n_embd, use_bias = self.config.use_bias, varw = self.config.varw_scale, dtype = self.config.dtype)
        self.fc_proj = self.Dense(self.config.n_embd, use_bias = self.config.use_bias, varw = self.config.varw_scale, dtype = self.config.dtype) # Either use variance_scaling(self.config.varw_scale / self.config.n_layers) or branch scaling

    def __call__(self, x: Array) -> Array:
        # B, T, n
        x = self.fc(x) # (B, T, ffwd_upscale * n)
        x = nn.gelu(x, approximate = True)
        x = self.fc_proj(x) # # (B, T, n)
        return x

""" MoE block """

# Adopted from https://github.com/huggingface/transformers/blob/4b822560a1dfd5d63c985ecf9a3c0aae0a4feeee/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py
# NOTE: Didnt include the common router

class Qwen_Block(nn.Module):
    config: dict
    Dense: nn.Module
    Readout: nn.Module

    def setup(self):
        self.gate = self.Readout(self.config.num_experts, use_bias = self.config.use_bias)
        self.experts = [MLP_Block(self.config, Dense = self.Dense) for _ in range(self.config.num_experts)]
        
    def __call__(self, x):
        B, T, n = x.shape
        x = x.reshape(-1, n)  # (BT, n)
        router_logits = self.gate(x)  # (BT, num_experts)
        routing_probs = jax.nn.softmax(router_logits, axis=-1)  # (BT, num_experts)
        routing_probs, selected_experts = jax.lax.top_k(routing_probs, self.config.top_k)  # (BT, top_k)

        if self.config.norm_topk_prob:
            routing_probs /= routing_probs.sum(axis=-1, keepdims=True)  # (BT, k)
        
        final_outputs = jnp.zeros(shape = (B*T, n))
        # create expert mask
        expert_mask = jax.nn.one_hot(selected_experts, num_classes=self.config.num_experts)  # (BT, top_k, num_experts)
        expert_mask = jnp.transpose(expert_mask, (2, 1, 0))  # (num_experts, top_k, BT)

        for expert_idx, expert_layer in enumerate(self.experts):
            expert_active = expert_mask[expert_idx]  # (top_k, BT)            
            # Instead of nonzero, we use where and cumsum to create indices
            mask = expert_active.any(axis=0)
            cumsum = jnp.cumsum(mask) - 1
            indices = jnp.where(mask, cumsum, -1)
            
            selected_inputs = jnp.where(mask[:, None], x, 0.0)
            expert_outputs = expert_layer(selected_inputs)
            
            # Compute the correct routing probabilities
            routing_probs_masked = jnp.where(mask[:, None],  routing_probs[jnp.arange(B*T), jnp.argmax(expert_active, axis=0)][:, None], 0.0)
            
            expert_outputs *= routing_probs_masked
            
            final_outputs = final_outputs.at[indices].add(expert_outputs)

        return final_outputs.reshape(B, T, n)
    

""" Multi-head attention Block """

class MultiHeadAttention(nn.Module):
    config: GPTConfig
    Dense: nn.Module

    def setup(self):
        # attention matrices
        self.attn = self.Dense(3 * self.config.n_embd, use_bias = self.config.use_bias, dtype = self.config.dtype, varw = self.config.varw_scale)
        # projection for multi-head attention
        self.proj = self.Dense(self.config.n_embd, use_bias = self.config.use_bias, dtype = self.config.dtype, varw = self.config.varw_scale)
        
    def __call__(self, x: Array) -> Array:

        B, T, n = x.shape  # B, T, n

        # key, query and value 
        q, k, v = jnp.split(self.attn(x), 3, axis = 2) # B, T, 3n

        # reshape them for multi-head attention
        q = q.reshape(B, T, self.config.n_head, n // self.config.n_head).transpose(0, 2, 1, 3)  # (B, n_head, T, d)
        k = k.reshape(B, T, self.config.n_head, n // self.config.n_head).transpose(0, 2, 1, 3)  # (B, n_head, T, d)
        v = v.reshape(B, T, self.config.n_head, n // self.config.n_head).transpose(0, 2, 1, 3)  # (B, n_head, T, d)

        # creates a lower triangular matrix of size (T, T)
        mask = jnp.tril(jnp.ones((1, 1, T, T))).astype(bool)

        # Attention
        att = q @ k.transpose(0, 1, 3, 2) / k.shape[-1]**self.config.attn_scale  # (B, n_head, T, T)
        # apply the mask by setting the upper triangular matrix to negative infinity
        att = jnp.where(mask, att, jnp.finfo(self.config.dtype).min) # np.where(condition, x, y) # jnp.finfo(self.config.dtype).min gives something close to negative infinity
        att = nn.softmax(att, axis = -1)
        y = att @ v  # (B, n_head, T, T) x (B, n_head, T, hs) -> (B, n_head, T, d)
        # combine all the heads into one
        y = y.transpose(0, 2, 1, 3).reshape(B, T, n) # (B, T, n_head, d) -> (B, T, n)
        # apply an affine transformation to mix up computations from different heads
        y = self.proj(y) 
        return y


""" GPT BLOCKs """

class MoEBlock(nn.Module):
    " Pre LN GPT Block "
    config: GPTConfig
    Dense: nn.Module
    Readout: nn.Module

    def setup(self):
        self.ln_1 = nn.LayerNorm(epsilon = 1e-5, dtype = self.config.dtype, use_bias = self.config.use_bias)
        self.attn = MultiHeadAttention(self.config, Dense = self.Dense)
        self.ln_2 = nn.LayerNorm(epsilon = 1e-5, dtype = self.config.dtype, use_bias = self.config.use_bias)
        self.moe = Qwen_Block(self.config, Dense = self.Dense, Readout = self.Readout)
        self.scale = 1.0 / ((2*self.config.n_blocks)**self.config.resd_scale)

    def __call__(self, x: Array) -> Array:
        x = x + self.scale * self.attn(self.ln_1(x)) # Apply scaled attention block
        x = x + self.scale * self.moe(self.ln_2(x)) # Apply scaled MLP block
        return x
    
class PreLNBlock(nn.Module):
    " Pre LN GPT Block "
    config: GPTConfig
    Dense: nn.Module

    def setup(self):
        self.ln_1 = nn.LayerNorm(epsilon = 1e-5, dtype = self.config.dtype, use_bias = self.config.use_bias)
        self.attn = MultiHeadAttention(self.config, Dense = self.Dense)
        self.ln_2 = nn.LayerNorm(epsilon = 1e-5, dtype = self.config.dtype, use_bias = self.config.use_bias)
        self.mlp = MLP_Block(self.config, Dense = self.Dense)
        self.scale = 1.0 / ((2*self.config.n_blocks)**self.config.resd_scale)

    def __call__(self, x: Array) -> Array:
        x = x + self.scale * self.attn(self.ln_1(x)) # Apply scaled attention block
        x = x + self.scale * self.mlp(self.ln_2(x)) # Apply scaled MLP block
        return x
    
class PostLNBlock(nn.Module):
    " Post LN GPT Block "
    config: GPTConfig
    Dense: nn.Module

    def setup(self):
        self.ln_1 = nn.LayerNorm(epsilon = 1e-5, dtype = self.config.dtype, use_bias = self.config.use_bias)
        self.attn = MultiHeadAttention(self.config, Dense = self.Dense)
        self.ln_2 = nn.LayerNorm(epsilon = 1e-5, dtype = self.config.dtype, use_bias = self.config.use_bias)
        self.mlp = MLP_Block(self.config, Dense = self.Dense)
        self.scale = 1.0 / ((2*self.config.n_blocks)**self.config.resd_scale)

    def __call__(self, x: Array) -> Array:
        x = self.ln_1(x + self.scale * self.attn(x)) # Apply scaled attention block
        x = self.ln_2(x + self.scale * self.mlp(x)) # Apply scaled MLP block
        return x
    
""" GPT MODELS """


class MoEGPT(nn.Module):
    """ MoE GPT """
    config: GPTConfig
    Dense: nn.Module = SimpleDense
    Readout: nn.Module = SimpleDense

    def setup(self):
        # input embedding
        self.tok_embd = nn.Embed(num_embeddings = self.config.vocab_size, features = self.config.n_embd, dtype = self.config.dtype, embedding_init = nn.initializers.normal(1.0))
        # positional embedding
        # self.pos_embd = nn.Embed(num_embeddings = self.config.context_length, features = self.config.n_embd, dtype = self.config.dtype, embedding_init = nn.initializers.normal(1.0)) # emmbedding is a map from integers to vectors
        self.pos_embd = SinusoidalPositionEmbed(self.config)
        # n_blocks GPT blocks 
        self.blocks = [MoEBlock(config = self.config, Dense = self.Dense, Readout = self.Readout) for _ in range(self.config.n_blocks)]
        # final Layer Norm
        self.ln_f = nn.LayerNorm(epsilon = 1e-5, dtype = self.config.dtype, use_bias = self.config.use_bias)
        # final read out layer
        self.lm_head = self.Readout(self.config.vocab_size, use_bias = self.config.use_bias, dtype = self.config.dtype, varw = 1.0)
        
    def __call__(self, tokens: Array) -> Array:
        # input embeddings
        T = tokens.shape[-1]
        token_embds = self.tok_embd(tokens) # (B, T, n_embd)
        pos_embds = self.pos_embd(tokens.shape[1])
        x = token_embds + pos_embds  #  (B, T, n_embd)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x    

class MoEGPT_muP(nn.Module):
    """ MoE GPT muP"""
    config: GPTConfig
    Dense: nn.Module = muDense
    Readout: nn.Module = muReadout

    def setup(self):
        # input embedding
        self.tok_embd = nn.Embed(num_embeddings = self.config.vocab_size, features = self.config.n_embd, dtype = self.config.dtype, embedding_init = nn.initializers.normal(1.0))
        # positional embedding
        # self.pos_embd = nn.Embed(num_embeddings = self.config.context_length, features = self.config.n_embd, dtype = self.config.dtype, embedding_init = nn.initializers.normal(1.0)) # emmbedding is a map from integers to vectors
        self.pos_embd = SinusoidalPositionEmbed(self.config)
        # n_blocks GPT blocks 
        self.blocks = [MoEBlock(config = self.config, Dense = self.Dense, Readout = self.Readout) for _ in range(self.config.n_blocks)]
        # final Layer Norm
        self.ln_f = nn.LayerNorm(epsilon = 1e-5, dtype = self.config.dtype, use_bias = self.config.use_bias)
        # final read out layer
        self.lm_head = self.Readout(self.config.vocab_size, use_bias = self.config.use_bias, varw = 1.0)
            
    def __call__(self, tokens: Array) -> Array:
        # input embeddings
        T = tokens.shape[-1]
        token_embds = self.tok_embd(tokens) # (B, T, n_embd)
        pos_embds = self.pos_embd(tokens.shape[1])
        x = token_embds + pos_embds  #  (B, T, n_embd)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x    

class PreLNGPT(nn.Module):
    """ Pre LN GPT """
    config: GPTConfig
    Dense: nn.Module = SimpleDense
    Readout: nn.Module = SimpleDense

    def setup(self):
        # input embedding
        self.tok_embd = nn.Embed(num_embeddings = self.config.vocab_size, features = self.config.n_embd, dtype = self.config.dtype, embedding_init = nn.initializers.normal(1.0))
        # positional embedding
        # self.pos_embd = nn.Embed(num_embeddings = self.config.context_length, features = self.config.n_embd, dtype = self.config.dtype, embedding_init = nn.initializers.normal(1.0)) # emmbedding is a map from integers to vectors
        self.pos_embd = SinusoidalPositionEmbed(self.config)
        # n_blocks GPT blocks 
        self.blocks = [PreLNBlock(self.config, Dense = self.Dense) for _ in range(self.config.n_blocks)]
        # final Layer Norm
        self.ln_f = nn.LayerNorm(epsilon = 1e-5, dtype = self.config.dtype, use_bias = self.config.use_bias)
        # final read out layer
        self.lm_head = self.Readout(self.config.vocab_size, use_bias = self.config.use_bias, dtype = self.config.dtype, varw = 1.0)
        
    def __call__(self, tokens: Array) -> Array:
        # input embeddings
        T = tokens.shape[-1]
        token_embds = self.tok_embd(tokens) # (B, T, n_embd)
        pos_embds = self.pos_embd(tokens.shape[1])
        x = token_embds + pos_embds  #  (B, T, n_embd)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = self.lm_head(x) 
        return x    

class PreLNGPT_muP(nn.Module):
    """ Pre LN GPT """
    config: GPTConfig
    Dense: nn.Module = muDense
    Readout: nn.Module = muReadout

    def setup(self):
        # input embedding
        self.tok_embd = nn.Embed(num_embeddings = self.config.vocab_size, features = self.config.n_embd, dtype = self.config.dtype, embedding_init = nn.initializers.normal(1.0))
        # positional embedding
        # self.pos_embd = nn.Embed(num_embeddings = self.config.context_length, features = self.config.n_embd, dtype = self.config.dtype, embedding_init = nn.initializers.normal(1.0)) # emmbedding is a map from integers to vectors
        self.pos_embd = SinusoidalPositionEmbed(self.config)
        # n_blocks GPT blocks 
        self.blocks = [PreLNBlock(self.config, Dense = self.Dense) for _ in range(self.config.n_blocks)]
        # final Layer Norm
        self.ln_f = nn.LayerNorm(epsilon = 1e-5, dtype = self.config.dtype, use_bias = self.config.use_bias)
        # final read out layer
        self.lm_head = self.Readout(self.config.vocab_size, use_bias = self.config.use_bias, dtype = self.config.dtype, varw = 1.0)
        
    def __call__(self, tokens: Array) -> Array:
        # input embeddings
        T = tokens.shape[-1]
        token_embds = self.tok_embd(tokens) # (B, T, n_embd)
        pos_embds = self.pos_embd(tokens.shape[1])
        x = token_embds + pos_embds  #  (B, T, n_embd)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = self.lm_head(x) 
        return x    


class PreLNGPTs(nn.Module):
    """ GPTs: Pre LN GPT but without the last LN """
    config: GPTConfig
    Dense: nn.Module = SimpleDense
    Readout: nn.Module = SimpleDense

    def setup(self):
        # input embedding
        self.tok_embd = nn.Embed(num_embeddings = self.config.vocab_size, features = self.config.n_embd, dtype = self.config.dtype, embedding_init = nn.initializers.normal(1.0))
        # positional embedding
        # self.pos_embd = nn.Embed(num_embeddings = self.config.context_length, features = self.config.n_embd, dtype = self.config.dtype, embedding_init = nn.initializers.normal(1.0)) # emmbedding is a map from integers to vectors
        self.pos_embd = SinusoidalPositionEmbed(self.config)
        # n_blocks GPT blocks 
        self.blocks = [PreLNBlock(self.config, Dense = self.Dense) for _ in range(self.config.n_blocks)]
        # final read out layer
        self.lm_head = self.Readout(self.config.vocab_size, use_bias = self.config.use_bias, dtype = self.config.dtype, varw = 1.0)
        
    def __call__(self, tokens: Array) -> Array:
        # input embeddings
        T = tokens.shape[-1]
        token_embds = self.tok_embd(tokens) # (B, T, n_embd)
        pos_embds = self.pos_embd(tokens.shape[1])
        x = token_embds + pos_embds  #  (B, T, n_embd)
        for block in self.blocks:
            x = block(x)
        x = self.lm_head(x) 
        return x    

class PostLNGPT(nn.Module):
    """ Post LN GPT """
    config: GPTConfig
    Dense: nn.Module = SimpleDense
    Readout: nn.Module = SimpleDense

    def setup(self):
        # input embedding
        self.tok_embd = nn.Embed(num_embeddings = self.config.vocab_size, features = self.config.n_embd, dtype = self.config.dtype, embedding_init = nn.initializers.normal(1.0))
        # positional embedding
        # self.pos_embd = nn.Embed(num_embeddings = self.config.context_length, features = self.config.n_embd, dtype = self.config.dtype, embedding_init = nn.initializers.normal(1.0)) # emmbedding is a map from integers to vectors
        self.pos_embd = SinusoidalPositionEmbed(self.config)
        # n_blocks GPT blocks 
        self.blocks = [PostLNBlock(self.config, Dense = self.Dense) for _ in range(self.config.n_blocks)]
        # final read out layer
        self.lm_head = self.Readout(self.config.vocab_size, use_bias = self.config.use_bias, dtype = self.config.dtype, varw = 1.0)
        
    def __call__(self, tokens: Array) -> Array:
        # input embeddings
        T = tokens.shape[-1]
        token_embds = self.tok_embd(tokens) # (B, T, n_embd)
        pos_embds = self.pos_embd(tokens.shape[1])
        x = token_embds + pos_embds  #  (B, T, n_embd)
        for block in self.blocks:
            x = block(x)
        x = self.lm_head(x)
        return x    

