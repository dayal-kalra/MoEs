from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents

from tqdm import tqdm
import numpy as np
import os


def bpe_tokenizer(files, ds_name, vocab_size, load_from_cache = True):
    """ Takes the files and trains the BPE tokenizer with vocab_size 
        1. Start with all the characters present in the training corpus as tokens
        2. Identify the most common pair of tokens and merge it into one token
        3. Repeat until the vocab_size is reached 
        Default vocab size for GPT2: 50,257
    """

    unk_token = '[UNK]'  # token for unknown words
    spl_tokens = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]']  # special tokens are not used training

    path = f'/nfshomes/dayal/tokenizers/trained_tokenizer_bpe-{ds_name}_vcb{vocab_size}.json'

    # if a trained tokenizer exists,then just load it
    if os.path.isfile(path) and load_from_cache:
        tokenizer = Tokenizer.from_file(path) # load from file
        
    else: # train a tokenizer from scratch
        # instantiate a Tokenizer using BPE 
        tokenizer = Tokenizer(BPE(unk_token = unk_token))

        # instantiate a normalizer; a normalizer cleans up the raw string. For example, stripping whitespace, removing accented characters, lowercasing
        normalizer = normalizers.Sequence([NFD(), StripAccents(), Lowercase()])
        tokenizer.normalizer = normalizer

        # pre tokenizer split inputs into words
        #  Without a pre-tokenizer, we might get tokens that overlap several words: for instance we could get an "it is" token since those two words often appear next to each other. Using a pre-tokenizer will ensure no token is bigger than a word returned by the pre-tokenizer. Here we want to train a subword BPE tokenizer, and we will use the easiest pre-tokenizer possible by splitting on whitespace.
        tokenizer.pre_tokenizer = Whitespace()

        # instantiate a BPE trainer
        trainer = BpeTrainer(vocab_size = vocab_size, special_tokens = spl_tokens)

        # training the tokenzier
        tokenizer.train(files, trainer)

        # save the tokenizer
        tokenizer.save(path)
        
    return tokenizer


def bert_tokenizer(files, dataset_name, vocab_size = 30522, load_from_cache = True):
    """ Takes the files and trains the tokenizer """

    unk_token = '[UNK]'  # token for unknown words
    spl_tokens = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]']  # special tokens are not used training

    path = f'tokenizers/trained_tokenizer_bert-{dataset_name}_vcb{vocab_size}.json'
    # if a trained tokenizer exists,then just load it
    if os.path.isfile(path) and load_from_cache:
        tokenizer = Tokenizer.from_file(path)

    else: # train a tokenizer from scratch
    
        # instantiate a Tokenizer using BPE 
        tokenizer = Tokenizer(WordPiece(unk_token = unk_token))

        # instantiate a normalizer; a normalizer cleans up the raw string. For example, stripping whitespace, removing accented characters, lowercasing
        normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()]) # strips accents
        tokenizer.normalizer = normalizer

        # instantiate a Wordpiece trainer
        trainer = WordPieceTrainer(vocab_size = vocab_size, special_tokens = spl_tokens)

        # Without a pre-tokenizer that will split our inputs into words,
        # we might get tokens that overlap several words: 
        # for instance we could get an "it is" token since those two words often appear next to each other. 
        # Using a pre-tokenizer will ensure no token is bigger than a word returned by the pre-tokenizer.
        tokenizer.pre_tokenizer = Whitespace()
        # pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)]) # multiple pre tokenizers

        # post processing
        tokenizer.post_processor = TemplateProcessing(
            single = "[CLS] $A [SEP]", # single sentences should have the form; $A is the sentence
            pair = "[CLS] $A [SEP] $B:1 [SEP]:1", # :1 after $B and the second [SEP] signifies that the tokens from the second sentence ($B) and the trailing [SEP] token should have a segment ID of 1. Segment IDs are used by models like BERT to distinguish between the first and second sentences in a pair. By default, tokens receive a segment ID of 0, which is why there's no :0 after $A
            special_tokens=[("[CLS]", spl_tokens.index("[CLS]")), ("[SEP]", spl_tokens.index("[SEP]")),],
        )

        # padding disabled currently
        #tokenizer.enable_padding(pad_id = spl_tokens.index("[PAD]"), pad_token = "[PAD]", length = pad_length+1)

        # training the tokenzier
        tokenizer.train(files, trainer)

        # save the tokenizer
        tokenizer.save(path)

    return tokenizer

def batch_encode(tokenizer, sentences, batch_size=100):
    batch_tokens = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        encoded = [tokenizer.encode(sentence).ids for sentence in batch]
        batch_tokens.extend(encoded)
        # Optionally, you can add memory management here
    return batch_tokens

