# custom utils
from utils.tokenizers_utils import bpe_tokenizer
import utils.train_utils as train
import utils.lms as lms
import utils.data_utils as data
import utils.loss_utils as loss_utils
import utils.schedules_utils as schedules_utils
import utils.optim_utils as optim

# jax, flax and optax imports
import jax
from jax import numpy as jnp
import optax
from flax import linen as nn

# for typing
from typing import Tuple

#usual imports
import numpy as np
import pandas as pd
import argparse

# for deterministic gpu computations
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'

def create_train_state(init_key, config: argparse.ArgumentParser, tokens: Tuple):
    
    inputs = tokens[:config.batch_size, :-1]

    # create model
    gpt2config = lms.GPTConfig(vocab_size = config.vocab_size, cntxt_len = config.cntxt_len, n_blocks = config.n_blocks, n_head = config.n_head, n_embd = config.n_embd, ffwd_upscale = config.ffwd_upscale, varw_scale = config.varw_scale, resd_scale = config.resd_scale, readout_scale = config.readout_scale, attn_scale = config.attn_scale, use_bias = config.use_bias)
    model = models[config.model_name](gpt2config)
    
    # initialize using the init seed
    init_params = model.init(init_key, inputs)['params']
    embd_params = config.n_embd * config.vocab_size

    # debugging: check shapes and norms
    shapes = jax.tree_util.tree_map(lambda x: x.shape, init_params)
    print(shapes)
    norms = jax.tree_util.tree_map(lambda x: jnp.var(x), init_params)
    # print(norms)

    # count the number of parameters
    num_params = lms.count_parameters(init_params)
    print(f'The model has {num_params/1e6:0.4f}M parameters with {embd_params/1e06:0.4f}M embedding parameters')

    # create an optimizer
    opt = optax.inject_hyperparams(optim.custom_sgdm)(learning_rate = config.lr_init, momentum = config.momentum)

    # create a train state
    state = train.TrainState.create(apply_fn = model.apply, params = init_params, opt = opt)

    return state, (num_params, embd_params)

def compute_metrics_dataset(state, loader, loss_function, num_examples, batch_size):
    """ Description: Estimates the loss of a batched data stream """
    total_loss = 0
    num_batches = train.estimate_num_batches(num_examples, batch_size)
    for batch_ix in range(num_batches):
        tokens = next(loader)
        tokens = jnp.asarray(tokens)
        inputs, labels = tokens[:, :-1], tokens[:, 1:]
        mask = jnp.ones_like(labels)
        batch = inputs, labels, mask
        loss = train.compute_loss(state, state.params, batch, loss_function)
        total_loss += loss
    ds_loss = total_loss / num_batches
    return ds_loss

def compute_sharpness_dataset(state, loader, loss_function, vs, num_batches):
    total_sharpness = 0
    for batch_idx in range(num_batches):
        tokens = next(loader)
        tokens = jnp.asarray(tokens)
        inputs, labels = tokens[:, :-1], tokens[:, 1:]
        mask = jnp.ones_like(labels)
        batch = inputs, labels, mask
        eigs, eigvs, n_iter = train.hessian_lobpcg_step(state, batch, loss_function, vs, m_iter = 1000, tol = 1e-09)
        sharpness = eigs.squeeze()
        total_sharpness += sharpness
    total_sharpness /= num_batches
    return total_sharpness


def train_and_evaluate(config: argparse.ArgumentParser, train_ds, valid_ds, test_ds):
    "train model according the config"
    
    init_key = jax.random.PRNGKey(config.main_seed)    
   
    # create a train state
    state, (num_params, embd_params) = create_train_state(init_key, config, train_ds)
     
    # create train and test batches for measurements: measure batches are called train_batches and val_batches; training batches are called batches
    seed = config.minibatch_seed
    train_loader = train.data_stream(seed, train_ds, config.batch_size)
    val_loader = train.data_stream(seed, valid_ds, config.batch_size)
    test_loader = train.data_stream(seed, test_ds, config.batch_size)

    # store training results
    eval_results = list()
    train_results = list()
    
    ### TRAINING PHASE
    
    divergence = False    
    running_loss = 0

    num_steps_per_epoch = train.estimate_num_batches(config.num_train, config.batch_size)
    print(f'Number of steps per epoch: {num_steps_per_epoch}')

    state.update_learning_rate(learning_rate = config.lr_init)
    config.lr_min = config.lr_trgt / 10.0
    lr_step = config.lr_init
    

    for step in range(config.num_steps):

        epoch = (step // config.num_batches) + 1
        cosine_step = state.step - config.warmup_steps + 1

        tokens = next(train_loader)
        tokens = jnp.asarray(tokens)
        inputs, labels = tokens[:, :-1], tokens[:, 1:]
        mask = jnp.ones_like(labels)
        batch = inputs, labels, mask

        # # update the learning rate in the warmup phase
        if step < config.warmup_steps:
            lr_step = schedules_utils.polynomial_warmup(state.step+1, config.lr_init, config.lr_trgt, config.warmup_steps, exponent = config.warmup_exponent) # state.step + 1 used because there is not training step yet
        else:
            lr_step = schedules_utils.cosine_decay_schedule(cosine_step+1, config.lr_trgt, config.lr_min, config.num_steps - config.warmup_steps + 1, exponent = config.decay_exponent)
        
        # update the learning rate
        state.update_learning_rate(learning_rate = lr_step)
   
        # train for one step
        state, grads_step, logits_step, loss_step = train.train_step(state, batch, config.loss_fn)
        # estimate grads norm
        flat_grads, rebuild_fn = jax.flatten_util.ravel_pytree(grads_step)
        grads_norm_step = jnp.linalg.norm(flat_grads)

        # estimate logits norm
        logits_norm_step = jnp.linalg.norm(logits_step)

        print(f't: {state.step}, epoch: {epoch}, lr: {lr_step:0.4f}, loss: {loss_step: 0.4f}')
        result = np.array([state.step, epoch, lr_step, loss_step, grads_norm_step, logits_norm_step])
        train_results.append(result)

        # increment the running loss
        running_loss += loss_step

        #check for divergence
        if (jnp.isnan(loss_step) or jnp.isinf(loss_step)): divergence = True; break

        if state.step % config.eval_interval == 0:
            train_loss_step = running_loss / config.eval_interval; running_loss = 0

            valid_loss_step = compute_metrics_dataset(state, val_loader, config.loss_fn, config.num_valid, config.batch_size)
            test_loss_step = compute_metrics_dataset(state, test_loader, config.loss_fn, config.num_test, config.batch_size)
            print(f't: {state.step}, lr_step: {lr_step:0.4f}, training loss: {train_loss_step:0.4f}, valid_loss: {valid_loss_step:0.4f}, test_loss: {test_loss_step:0.4f}')
            result = np.asarray([state.step, epoch, lr_step, train_loss_step, valid_loss_step, test_loss_step])
            eval_results.append(result)

    train_results = np.asarray(train_results)
    eval_results = np.asarray(eval_results)
    return divergence, train_results, eval_results, num_params, embd_params


models = {'moegpt': lms.MoEGPT, 'moegpt_mup': lms.MoEGPT_muP, 'pregpt': lms.PreLNGPT, 'pregpt_mup': lms.PreLNGPT_muP, 'pregpts': lms.PreLNGPTs, 'postgpt':lms.PostLNGPT}
losses = {'mse': loss_utils.mse_loss, 'xent': loss_utils.cross_entropy_loss, 'masked_xent': loss_utils.masked_cross_entropy_loss}

parser = argparse.ArgumentParser(description = 'Hyperparameters')
parser.add_argument('--cluster', type = str, default = 'nexus')
parser.add_argument('--main_seed', type = int, default = 1)
### Dataset hyperparams
parser.add_argument('--ds_name', type = str, default = 'wikitext-2')
parser.add_argument('--ds_dir', type = str, default = '/home/dayal/scratch.mzms/datasets')
parser.add_argument('--vocab_size', type = int, default = 4096) # default 50,257.
parser.add_argument('--num_tokens', type = int, default = -1) # 32, 768 can be fit into A100 with B = 256 and T = 128

### Architectural hyperparams
# https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
parser.add_argument('--model_name', type = str, default = 'pregpt') 
parser.add_argument('--cntxt_len', type = int, default = 64) # 1024
parser.add_argument('--n_blocks', type = int, default = 4) # 12 by default
parser.add_argument('--n_head', type = int, default = 4) # 8 by default
parser.add_argument('--n_embd', type = int, default = 128) # 768 by default source 
parser.add_argument('--ffwd_upscale', type = int, default = 4)
parser.add_argument('--resd_scale', type = float, default = 0.0)
parser.add_argument('--readout_scale', type = float, default = 0.0)
parser.add_argument('--attn_scale', type = float, default = 0.5)
parser.add_argument('--varw_scale', type = float, default = 2.0)
parser.add_argument('--act_name', type = str, default = 'gelu') # TODO: implement 
parser.add_argument('--bias', type = str, default = 'False') 

### Loss function
parser.add_argument('--loss_name', type = str, default = 'masked_xent') # create your mask during forward pass

### Optimization hyperparams
parser.add_argument('--augment', type = str, default = 'True')
parser.add_argument('--opt_name', type = str, default = 'base_sgd')
parser.add_argument('--sgd_seed', type = int, default = 1)
parser.add_argument('--warmup_steps', type = int, default = 1)
parser.add_argument('--warmup_exponent', type = float, default = 1.0) # exponent for warmup
parser.add_argument('--decay_exponent', type = float, default = 0.0) # exponent for decay
parser.add_argument('--num_steps', type = int, default = 1_000)
parser.add_argument('--lr_start', type = float, default = 1e-04)
parser.add_argument('--lr_init', type = float, default = 0.0)
parser.add_argument('--lr_trgt', type = float, default = 1e-01)
parser.add_argument('--momentum', type = float, default = 0.0)
# mini batch hyperparams
parser.add_argument('--minibatch_seed', type = int, default = 1)
parser.add_argument('--batch_size', type = int, default = 16) # 16 by default
### Evaluation hyperparams
parser.add_argument('--eval_interval', type = int, default = 100)
parser.add_argument('--results_dir', type = str, default = 'results')
# Sharpness estimation
parser.add_argument('--topk', type = int, default = 1)
parser.add_argument('--sharpness_method', type = str, default = 'lobpcg')
parser.add_argument('--measure_batches', type = int, default = 10)


save_dir = 'results'

config = parser.parse_args()

# setup loss function
config.loss_fn = losses[config.loss_name]

config.use_bias = True if config.bias == 'True' else False 

# directories for different clusters

if config.cluster == 'nexus':
    config.ds_dir = '/nfshomes/dayal/datasets'
elif config.cluster == 'zaratan':
    config.ds_dir = '/home/dayal/scratch.mzms/datasets'
else:
    config.ds_dir = 'datasets'

assert (config.n_embd % config.n_head == 0), 'Embedding dimension not divisble by the number of heads'

files = [f'{config.ds_dir}/{config.ds_name}/{config.ds_name}.{split}' for split in ['train']] # not using 'validation', 'test'
trained_tokenizer = bpe_tokenizer(files, config.ds_name, config.vocab_size, load_from_cache = True) # load_from_cache should be false if the context length is changed

config.vocab_size = trained_tokenizer.get_vocab_size()
PAD_TOKEN = trained_tokenizer.token_to_id("[PAD]")


train_ds, valid_ds, test_ds = data.load_dataset(config, trained_tokenizer)

config.num_train = len(train_ds)
config.num_valid = len(valid_ds)
config.num_test = len(test_ds)

config.num_batches = train.estimate_num_batches(config.num_train, config.batch_size)

print(config)

divergence = False
config.lr_trgt = 1e-02

while not divergence:

    train_path = f'{config.results_dir}/train_{config.ds_name}_T{config.num_tokens}_v{config.vocab_size}_{config.model_name}_cont{config.cntxt_len}_n{config.n_embd}_h{config.n_head}_d{config.n_blocks}_u{config.ffwd_upscale}_varw{config.varw_scale}_a{config.readout_scale}_b{config.resd_scale}_at{config.attn_scale}_bias{config.use_bias}_{config.act_name}_I{config.main_seed}_{config.loss_name}_{config.opt_name}_lr{config.lr_trgt:0.6f}_Twrm{config.warmup_steps}_T{config.num_steps}_B{config.batch_size}_m{config.momentum}.tab'
    eval_path = f'{config.results_dir}/eval_{config.ds_name}_T{config.num_tokens}_v{config.vocab_size}_{config.model_name}_cont{config.cntxt_len}_n{config.n_embd}_h{config.n_head}_d{config.n_blocks}_u{config.ffwd_upscale}_varw{config.varw_scale}_a{config.readout_scale}_b{config.resd_scale}_at{config.attn_scale}_bias{config.use_bias}_{config.act_name}_I{config.main_seed}_{config.loss_name}_{config.opt_name}_lr{config.lr_trgt:0.6f}_Twrm{config.warmup_steps}_T{config.num_steps}_B{config.batch_size}_m{config.momentum}.tab'
        
    
    divergence, train_results, eval_results, num_params, embd_params = train_and_evaluate(config, train_ds, valid_ds, test_ds)
    # save training data
    df_train = pd.DataFrame(train_results, columns = ['step', 'epoch', 'lr', 'train_loss', 'grads_norm', 'logits_norm'])
    df_train['num_params'] = num_params; df_train['embd_params'] = embd_params
    df_train.to_csv(train_path, sep = '\t')

    if not divergence:
        # save eval data
        df_eval = pd.DataFrame(eval_results, columns = ['step', 'epoch', 'lr', 'train_loss', 'valid_loss', 'test_loss'])
        df_eval['num_params'] = num_params; df_eval['embd_params'] = embd_params
        df_eval.to_csv(eval_path, sep = '\t')

    config.lr_trgt *= 2.0

        