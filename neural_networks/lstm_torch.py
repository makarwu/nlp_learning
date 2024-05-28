# LSTMs also work easily with PyTorch
from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from collections import defaultdict
from dataset import Dataset
from lstm import one_hot_encode_sequence
import matplotlib.pyplot as plt

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
vocab_size = len(word_to_idx)

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


class MyRecurrentNet(nn.Module):
    def __init__(self):
        super(MyRecurrentNet, self).__init__()

        # Recurrent layer
        self.lstm = nn.LSTM(input_size=vocab_size,
                            hidden_size = 50,
                            num_layers=1,
                            bidirectional=False)
        
        # Output layer
        self.l_out = nn.Linear(in_features=50,
                            out_features=vocab_size,
                            bias=False)
    
    def forward(self, x):
        # RNN returns output and last hidden state
        x, (h, c) = self.lstm(x)

        # Flatten the output for feed-forward layer
        x = x.view(-1, self.lstm.hidden_size)

        # Output layer
        x = self.l_out(x)

        return x

net = MyRecurrentNet()
print(net)
        
## Training Loop:
num_epochs = 200

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

# Track loss
training_loss, validation_loss = [], []

for i in range(num_epochs):

    epoch_training_loss = 0
    epoch_validation_loss = 0
    net.eval()

    for inputs, targets in validation_set:

        # One-hot encode input and target sequence
        inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size)
        targets_idx = [word_to_idx[word] for word in targets]

        # Convert the input to a tensor
        inputs_one_hot = torch.Tensor(inputs_one_hot).unsqueeze(0)
        inputs_one_hot = inputs_one_hot.permute(0, 2, 1)

        # Convert target to tensor
        targets_idx = torch.LongTensor(targets_idx).unsqueeze(0)

        # Zero out the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs_one_hot)

        #Compute loss
        loss = criterion(outputs, targets_idx.view(-1))

        # Update loss
        epoch_validation_loss += loss.item()
    
    net.train()

    for inputs, targets in training_set:

         # One-hot encode input and target sequence
        inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size)
        targets_idx = [word_to_idx[word] for word in targets]
        
        # Convert input to tensor
        inputs_one_hot = torch.Tensor(inputs_one_hot).unsqueeze(0)  # Add batch dimension
        inputs_one_hot = inputs_one_hot.permute(0, 2, 1)
        
        # Convert target to tensor
        targets_idx = torch.LongTensor(targets_idx).unsqueeze(0)  # Add batch dimension
        
        # Zero out gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs_one_hot)

        # Compute the loss
        loss = criterion(outputs, targets_idx.view(-1))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update loss
        epoch_training_loss += loss.item()

    # Save loss for plot
    training_loss.append(epoch_training_loss/len(training_set))
    validation_loss.append(epoch_validation_loss/len(validation_set))

    # Print loss every 10 epochs
    if i % 10 == 0:
        print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')

# Get first sentence in test set
inputs, targets = test_set[1]

# One-hot encode input and target sequence
inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size)
targets_idx = [word_to_idx[word] for word in targets]

# Convert input to tensor
inputs_one_hot = torch.Tensor(inputs_one_hot).unsqueeze(0)
inputs_one_hot = inputs_one_hot.permute(0, 2, 1)

# Convert target to tensor
targets_idx = torch.LongTensor(targets_idx).unsqueeze(0)

# Forward pass
outputs = net.forward(inputs_one_hot).data.numpy()

print('\nInput sequence:')
print(inputs)

print('\nTarget sequence:')
print(targets)

print('\nPredicted sequence:')
print([idx_to_word[np.argmax(output)] for output in outputs])

# Plot training and validation lossghggghhgjjzzggghhhhhgggjjkhkuzzthtrgdgd
epoch = np.arange(len(training_loss))
plt.figure()
plt.plot(epoch, training_loss, 'r', label='Training loss',)
plt.plot(epoch, validation_loss, 'b', label='Validation loss')
plt.legend()
plt.xlabel('Epoch'), plt.ylabel('NLL')
plt.show()