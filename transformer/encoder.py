from mhselfattention import MHSelfAttention
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, forward_expansion, dropout=0.1):
        super(TransformerEncoder, self).__init__()
    
        self.attention = MHSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attention_out = self.dropout(self.attention(x))
        x = self.norm1(x + attention_out)
        forward_out = self.dropout(self.feed_forward(x))
        out = self.norm2(x + forward_out)

        return out

