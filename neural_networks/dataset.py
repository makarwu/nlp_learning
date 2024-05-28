# Partitioning the Dataset
"""
Since we are doing next-word predictions (LSTM), our target sequence is simply the 
input sequence shifted by one word.
We can use PyTorch's Dataset class to build a simple dataset where we can 
easily retrieve (inputs, targets) pairs for each of our sequences.
"""
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        X = self.inputs[index]
        y = self.targets[index]
        return X, y
