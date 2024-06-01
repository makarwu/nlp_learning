import torch
import torch.nn as nn

# keep in mind: they used a positional encoding in the original paper
# here: learnable positional embedding is used:
class Embedding(nn.Module):
    def __init__(self, vocab_size, max_length, embed_dim, droput=0.1):
        super(Embedding, self).__init__()
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_length, embed_dim)
        self.dropout = nn.Dropout(droput)

    def forward(self, x):
        batch_size, seq_length, = x.shape
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        positions = torch.arange(0, seq_length).expand(
            batch_size, seq_length).to(device)
        embedding = self.word_embed(x) + self.pos_embed(positions)
        return self.dropout(embedding)
