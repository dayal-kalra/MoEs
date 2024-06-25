import utils.train_utils as train_utils
import optax


def compute_loss(state, batch, grads, loss_fn):
    # compute the next parameters
    updates, opt_state_next = state.opt.update(grads, state.opt_state, state.params)
    params_next = optax.apply_updates(state.params, updates)
    _, loss = train_utils.loss_step(state, batch, params_next, loss_fn)
    return state, loss

def exponential_search(config, state, batch, grads_init, loss_init, lr_zero = 1e-04):

    frwd_passes = 0
    lr_upper = lr_zero
    # set the learning rate
    state.update_learning_rate(learning_rate = lr_upper)
    # compute the initial loss
    state, loss_upper = compute_loss(state, batch, grads_init, config.loss_fn)
    # we use at least one forward pass to check if the loss is increasing with the default learning rate
    frwd_passes += 1   
    loss_lower = loss_upper

    print(f'lr_upper: {lr_upper:0.8f}, loss_upper: {loss_upper:0.8f}, loss_init: {loss_init:0.8f}')
    eps = 1e-03

    while loss_upper <= loss_init + eps:
        # update the previous loss
        loss_lower = loss_upper
        
        if lr_upper > config.lr_trgt:
            # set the learning rate to the target     
            lr_upper = config.lr_trgt
            # update the learning rate
            state.update_learning_rate(learning_rate = lr_upper)
            # compute the loss at next step
            state, loss_upper = compute_loss(state, batch, grads_init, config.loss_fn)
            frwd_passes += 1
            print(f'lr_upper: {lr_upper:0.6f}, loss_upper: {loss_upper:0.6f}, loss_init: {loss_init:0.6f}, frwd_passes: {frwd_passes}')
            break
        else:
            # 2x the learning rate
            lr_upper *= 2.0
            # update the learning rate
            state.update_learning_rate(learning_rate = lr_upper)
            # compute the loss at next step
            state, loss_upper = compute_loss(state, batch, grads_init, config.loss_fn)
            frwd_passes += 1
        
        print(f'lr_upper: {lr_upper:0.6f}, loss_upper: {loss_upper:0.6f}, loss_init: {loss_init:0.6f}, frwd_passes: {frwd_passes}')
        
    print(f'lr_upper: {lr_upper:0.6f}, loss_upper: {loss_upper:0.6f}, loss_init: {loss_init:0.6f}, frwd_passes: {frwd_passes}')

    lr_lower = lr_upper / 2.0
    return loss_lower, loss_upper, lr_lower, lr_upper, frwd_passes

def binary_search_loss(config, state, batch, grads_init, loss_init, loss_lower, loss_upper, lr_lower, lr_upper):
    frwd_passes = 0
    print(f'lr_lower: {lr_lower:0.6f}, lr_upper: {lr_upper:0.6f}, loss_lower: {loss_lower:0.6f}, loss_upper: {loss_upper:0.6f}, loss_init: {loss_init:0.6f}, frwd_passes: {frwd_passes}')
    while loss_upper > loss_init * (1 + config.eps):
        lr_mid = (lr_upper + lr_lower) / 2.0
        state.update_learning_rate(learning_rate = lr_mid)
        state, loss_mid = compute_loss(state, batch, grads_init, config.loss_fn)
        if loss_mid < loss_init:
            lr_lower = lr_mid
            loss_lower = loss_mid
        else:
            lr_upper = lr_mid
            loss_upper = loss_mid
        print(f'lr_lower: {lr_lower:0.6f}, lr_upper: {lr_upper:0.6f}, loss_lower: {loss_lower:0.6f}, loss_upper: {loss_upper:0.6f}, loss_init: {loss_init:0.6f}, frwd_passes: {frwd_passes}')

    return loss_lower, loss_upper, lr_lower, lr_upper, frwd_passes

def binary_search_lr(config, state, batch, grads_init, loss_init, loss_lower, loss_upper, lr_lower, lr_upper):
    frwd_passes = 0
    return loss_lower, loss_upper, lr_lower, lr_upper, frwd_passes

def search_instability(config, state, batch, lr_zero):
    """
    Inputs: 
        config: config file
        state: model state
        batch: 
        lr_zero: initial guess for learning rate search
    """

    xtra_frwd_passes = 0 # keeps track of extra forward passes

    # estimate the initial loss and gradients; we need only the gradients at initialization
    # we do not count this as an extra forward pass because this information is used in training
    # an effecient implementation would utilize this information
    grads_init, loss_init = train_utils.grads_step(state, batch, config.loss_fn)

    print(f'Initiate exponential search')
    print(f'lr_zero: {config.lr_zero:0.6f}, loss_init: {loss_init:0.4f}')

    ## Exponential search for the learning rate upper bound
    
    loss_lower, loss_upper, lr_lower, lr_upper, frwd_passes = exponential_search(config, state, batch, grads_init, loss_init,  lr_zero = config.lr_zero)
    xtra_frwd_passes += frwd_passes

    if loss_upper > loss_init > loss_lower:  # biserction search only if the following inequality holds

        print(f'Initiate binary search')
        ## Binary search / bisection method
        loss_lower, loss_upper, lr_lower, lr_upper, frwd_passes = binary_search_loss(config, state, batch, grads_init, loss_init, loss_lower, loss_upper, lr_lower, lr_upper)
        xtra_frwd_passes += frwd_passes
        print(f'loss_lower: {loss_lower:0.4f}, loss_upper: {loss_upper:0.4f}, lr_lower: {lr_lower:0.4f}, lr_upper {lr_upper:0.4f}, frwd_passes: {xtra_frwd_passes}')

    return lr_lower, lr_upper, xtra_frwd_passes