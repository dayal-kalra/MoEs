from tokenizers_utils import bpe_tokenizer, bert_tokenizer

def load_dataset(files):
    """ Loads a language dataset """    
    datasets = []
    for file in files:
        with open(file, 'r', encoding = 'utf-8') as fi:
            datasets.append(fi.read())
    return datasets

tokenizers = {'bpe': bpe_tokenizer, 'bert': bert_tokenizer}

dataset_name = 'wikitext'
dataset_dir = '/nfshomes/dayal/llms/datasets/'

files = [f'{dataset_dir}/{dataset_name}/{dataset_name}.{split}' for split in ['train', 'valid', 'test']]
trained_tokenizer = bert_tokenizer(files, dataset_name, padding = True)

train_ds, valid_ds, test_ds = load_dataset(files)

output = trained_tokenizer.encode("Hello, y'all! How are you 😁 ?")

print('*'*100)
print(output.tokens)
print(output.ids)
print(output.type_ids)
print('*'*100)

output = trained_tokenizer.encode("Hello, y'all!", "How are you 😁 ?")

print('*'*100)
print(output.tokens)
print(output.ids)
print(output.type_ids)
print('*'*100)

output = trained_tokenizer.encode_batch([["Hello, y'all!", "How are you 😁 ?"], ["Hello to you too!", "I'm fine, thank you!"]])


print('*'*100)
print(output[0].tokens)
print(output[0].ids)
print(output[0].type_ids)
print(output[0].attention_mask)

print('*'*100)

print(output[1].tokens)
print(output[1].ids)
print(output[1].type_ids)
print(output[1].attention_mask)
print(trained_tokenizer.decode(output[1].ids))

print('*'*100)



