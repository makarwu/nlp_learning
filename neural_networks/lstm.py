# LSTM - from scratch (Long Short Term Memory)
# 1. we generate a dataset
# 2. we generate an LSTM network with NumPy
# 3. we generate an LSTM network with PyTorch

""" 1. Generating a dataset
We will create a simple dataset that we can learn from.
We generate sequences of the form: 
a b EOS
a a b b EOS 
...
where EOS is a special character denoting the end of a sequence.
The task is to predict the next token. 
We process sequences in sequential manner. As such, the network will need
to learn that e.g. t b's and an EOS token will follow 5 a's.
"""

import numpy as np
from collections import defaultdict
from dataset import Dataset

np.random.seed(42)

def generate_dataset(num_sequences=2**8):

    samples = []
    for _ in range(num_sequences):
        num_tokens = np.random.randint(1, 12)
        sample = ['a'] * num_tokens + ['b'] * num_tokens + ['EOS']
        samples.append(sample)
    
    return samples

sequences = generate_dataset()
print('A single sample from the generated dataset:')
print(sequences[0])

""" Now, we need a function, which represents the tokens as indices.
To build a one-hot encoding, we need to assign each possible word in our
vocabulary an index:
We do that by creating two dictionaries: one that allows us to go from 
a given word to its corresponding index in our vocabulary, and one 
in reverse direction. 
If we try to access a word that does not exist in our vocabulary, it is
automatically replaced by the UNK token or its corresponding index.
"""

def sequences_to_dicts(sequences):

    # Python-magic to flatten a nested list
    flatten = lambda l: [item for sublist in l for item in sublist]
    # Flatten the dataset
    all_words = flatten(sequences)
    # Count the word occurences
    word_count = defaultdict(int)
    for word in flatten(sequences):
        word_count[word] += 1
    # Sort by frequency
    word_count = sorted(list(word_count.items()), key= lambda l: -l[1])
    # Create a list of all unique words
    unique_words = [item[0] for item in word_count]
    # Add UNK token to list of words
    unique_words.append('UNK')
    # Count number of sequences and number of unique words
    num_sentences, vocab_size = len(sequences), len(unique_words)
    # Create dictionaries so that we can go from word to index and back
    # If a word is not in our vocabulary, we assign it to token 'UNK'
    word_to_idx = defaultdict(lambda: vocab_size-1)
    idx_to_word = defaultdict(lambda: 'UNK')
    # Fill dictionaries
    for idx, word in enumerate(unique_words):
        word_to_idx[word] = idx
        idx_to_word[idx] = word
    
    return word_to_idx, idx_to_word, num_sentences, vocab_size

word_to_idx, idx_to_word, num_sequences, vocab_size = sequences_to_dicts(sequences)
print(f'We have {num_sequences} sentences and {len(word_to_idx)} unique tokens in our dataset (including UNK).\n')
print('The index of \'b\' is', word_to_idx['b'])
print(f'The word corresponding to index 1 is \'{idx_to_word[1]}\'')
#print(idx_to_word)
#print('----------')
#print(word_to_idx)
assert idx_to_word[word_to_idx['b']] == 'b', \
    'Consistency error: something went wrong in the conversion'

def create_datasets(sequences, dataset_class, p_train=0.8, p_val=0.1, p_test=0.1):
    # Define partition sizes
    num_train = int(len(sequences)*p_train)
    num_val = int(len(sequences)*p_val)
    num_test = int(len(sequences)*p_test)

    # Split sequences into partitions
    sequences_train = sequences[:num_train]
    sequences_val = sequences[num_train:num_train+num_val]
    sequences_test = sequences[-num_test:]

    def get_inputs_targets_from_sequences(sequences):
        # Define empty lists
        inputs, targets = [], []
        # Append inputs and targets s.t. both lists contain L-1 words of a sentence of length L
        # but targets are shifted right by one so that we can predict the next word
        for sequence in sequences:
            inputs.append(sequence[:-1])
            targets.append(sequence[1:])
            
        return inputs, targets

    # Get inputs and targets for each partition
    inputs_train, targets_train = get_inputs_targets_from_sequences(sequences_train)
    inputs_val, targets_val = get_inputs_targets_from_sequences(sequences_val)
    inputs_test, targets_test = get_inputs_targets_from_sequences(sequences_test)

    # Create datasets
    training_set = dataset_class(inputs_train, targets_train)
    validation_set = dataset_class(inputs_val, targets_val)
    test_set = dataset_class(inputs_test, targets_test)

    return training_set, validation_set, test_set
    

training_set, validation_set, test_set = create_datasets(sequences, Dataset)

print(f'We have {len(training_set)} samples in the training set.')
print(f'We have {len(validation_set)} samples in the validation set.')
print(f'We have {len(test_set)} samples in the test set.')
  

