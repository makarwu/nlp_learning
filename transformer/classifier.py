import torch.nn as nn
from embedding import Embedding
from encoder import TransformerEncoder

"""
Here, we build an End-to-End classifier
1. An embedding layer
2. A single transformer encoder layer
3. A fully-connected network as a linear classifier
"""
class Classifier(nn.Module):
    def __init__(self, vocab_size, max_length, embed_dim, num_heads,
                forward_expansion):
        super(Classifier, self).__init__()

        self.embedder = Embedding(vocab_size, max_length, embed_dim)
        self.encoder = TransformerEncoder(embed_dim, num_heads, forward_expansion)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        embedding = self.embedder(x)
        encoding = self.encoder(embedding)
        compact_encoding = encoding.max(dim=1)[0]
        out = self.fc(compact_encoding)
        return out
    
