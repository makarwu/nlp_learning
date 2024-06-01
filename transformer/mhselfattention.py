import torch.nn as nn
import torch

class MHSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MHSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (self.num_heads * self.head_dim == self.embed_dim), \
            'embed size must be devisible by number of heads'
        
        self.w_queries = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.w_keys = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.w_values = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.fc_out = nn.Linear(self.head_dim * self.num_heads, self.embed_dim)

    def forward(self, x):

        # shape of x = [batch_size, sentence_length, embedding_dim]
        batch_size = x.shape[0]
        sentence_len = x.shape[1]

        queries = self.w_queries(x).reshape(
            batch_size, sentence_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        keys = self.w_keys(x).reshape(
            batch_size, sentence_len, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        
        values = self.w_values(x).reshape(
            batch_size, sentence_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3) 
        
        """
        attention_score & attention dist explanation:
        - bijk (queries) bikl (keys) 'bmm ->' bijl (attention_scores)
        - where b is the batch size dimension, i the number of heads in
        multi-head attention
        - k represents the sequence length of the keys and values
        - l represents the feature dimension of the keys and values 

        overall: we compute the attention score by performing a batched
        matrix multiplication between queries and keys, summing over the 
        'k' dimension, and producing a result with dimensions (b, i, j, l).
        """
        attention_scores = torch.einsum('bijk,bikl->bijl', queries, keys)
        attention_dist = torch.softmax(attention_scores / 
                (self.embed_dim ** 0.5), dim=-1) 
        attention_out = torch.einsum('bijk,bikl->bijl', attention_dist, values)
        concatenated_out = attention_out.permute(0, 2, 1, 3).reshape(
            batch_size, sentence_len, self.embed_dim)
        
        return concatenated_out


