import numpy as np
import string
from nltk.corpus import stopwords 

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class word2vec(object):
    def __init__(self):
        self.N = 10
        self.X_train = []
        self.y_train = []
        self.window_size = 2
        self.alpha = 0.001
        self.words = []
        self.word_index = {}
    
    def initialize(self, V, data):
        self.V = V
        self.W = np.random.uniform(-0.8, 0.8, (self.V, self.N))
        self.W1 = np.random.uniform(-0.8, 0.8, (self.N, self.V))
        self.words = data
        for i in range(len(data)):
            self.word_index[data[i]] = i

    def forward(self, X):
        pass

    def backward(self, x, t):
        pass

    def train(self, epochs):
        pass

    def predict(self, word, number_of_predictions):
        pass
