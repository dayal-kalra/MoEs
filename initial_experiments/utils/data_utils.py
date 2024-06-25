import numpy as np

def load_dataset(config, tokenizer):
    """ Loads a language dataset given the file locations """    
    datasets = {}
    with open(f'{config.ds_dir}/{config.ds_name}/{config.ds_name}.train', 'r', encoding = 'utf-8') as fi:
        sentences = fi.readlines()
        tokens = tokenize_and_split_data(sentences, tokenizer, config.cntxt_len, config.num_tokens)
        datasets['train'] = tokens

    with open(f'{config.ds_dir}/{config.ds_name}/{config.ds_name}.valid', 'r', encoding = 'utf-8') as fi:
        sentences = fi.readlines()
        tokens = tokenize_and_split_data(sentences, tokenizer, config.cntxt_len)
        datasets['valid'] = tokens
    with open(f'{config.ds_dir}/{config.ds_name}/{config.ds_name}.test', 'r', encoding = 'utf-8') as fi:
        sentences = fi.readlines()
        tokens = tokenize_and_split_data(sentences, tokenizer, config.cntxt_len)
        datasets['test'] = tokens
    return datasets['train'], datasets['valid'], datasets['test']

def tokenize_and_split_data(sentences, tokenizer, cntxt_len, num_tokens = -1):

    token_ids = [np.array(tokenizer.encode(sentence).ids, dtype = int) for sentence in sentences]
    # Flatten the list of arrays into a single array using numpy.concatenate
    tokens = np.concatenate(token_ids)
    if num_tokens > 0:
        tokens = tokens[:num_tokens+1]
    
    print(f'Length of tokens: {tokens.shape}')

    num_complete_batches = len(tokens) // cntxt_len

    print(f'Num batches: {num_complete_batches}')

    last_token = tokens[num_complete_batches * cntxt_len]
    # Truncate tokens to fit complete batches and reshape
    tokens = tokens[:num_complete_batches * cntxt_len]
    tokens = tokens.reshape(-1, cntxt_len)

    first_tokens = np.roll(tokens[:, 0], -1) 
    first_tokens[-1] = last_token
    
    first_tokens = first_tokens.reshape(len(first_tokens), 1)

    tokens = np.append(tokens, first_tokens, axis = 1)

    # Setting the seed for reproducibility
    np.random.seed(42)  # You can choose any number as your seed
    indices = np.random.permutation(tokens.shape[0])
    tokens = tokens[indices]

    return tokens
    

    
