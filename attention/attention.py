"""Here we implement the Attention Mechanism from Scratch:

- The key of attention is determine which words are most important in a
specific context.

- We can think of self-attention as a mechanism that enhances the information
content of an input embedding by including information about the
input's context.

- In other words: the mechanism enables the model to weigh the importance
of different elements in an input sequence and dynamically adjust their
influence on the output.

- This is especially important for language processing tasks, where 
the meaning of a word can change based on its context within a sentence
or a document.

"""
import torch
import torch.nn.functional as F

#1: create a sentence embedding
sentence = 'Life is short, eat dessert first'

dc = {s:i for i, s in enumerate(sorted(sentence.replace(',', '').split()))}
print(dc)

#2: Next, we use this dictionary to assign an integer index to each word
sentence_int = torch.tensor([dc[s] for s in sentence.replace(',', '').split()])
print(sentence_int)

#3: Now, we have an integer-vector representation and can use an embedding
# layer to encode the inputs into a real-vector embedding.

torch.manual_seed(123)
embed = torch.nn.Embedding(6, 16)
embedded_sentence = embed(sentence_int).detach()

print(embedded_sentence)
print(embedded_sentence.shape) # 6x16 dimensional embedding


#4: Defining the weight matrices (wq, wk, wv)
d = embedded_sentence.shape[1]

d_q, d_k, d_v = 24, 24, 28 # keep in mind dq = dk

W_query = torch.nn.Parameter(torch.rand(d_q, d))
W_key = torch.nn.Parameter(torch.rand(d_k, d))
W_value = torch.nn.Parameter(torch.rand(d_v, d))

# Suppose we want to compute the attention vector for the 2nd input element:

x_2 = embedded_sentence[1]
query_2 = W_query.matmul(x_2)
key_2 = W_key.matmul(x_2)
value_2 = W_value.matmul(x_2)

print(query_2.shape)
print(key_2.shape)
print(value_2.shape)

# In general this looks like this:

keys = W_key.matmul(embedded_sentence.T).T
values = W_value.matmul(embedded_sentence.T).T

print("keys.shape: ", keys.shape)
print("values.shape : ", values.shape)

# Now, we want to compute the unnormalized attention weights w:

omega_2 = query_2.matmul(keys.T)
print(omega_2)

# Compute the normalized attention scores: 

attention_weights_2 = F.softmax(omega_2 / d_k**0.5, dim= 0)
print(attention_weights_2)

# Finally, the last step is to compute z, which is an attention weighted
# version of original query input x

context_vector_2 = attention_weights_2.matmul(values)

print(context_vector_2.shape)
print(context_vector_2)

# Let's implement multi-head Attention

h = 3
multihead_W_query = torch.nn.Parameter(torch.randn(h, d_q, d))
multihead_W_key = torch.nn.Parameter(torch.randn(h, d_k, d))
multihead_W_value = torch.nn.Parameter(torch.randn(h, d_v, d))

# Now, we can compute all the keys and values using torch.bmm() (batch matrix multiplication)
# But first we need to expand the input size by 3
stacked_inputs = embedded_sentence.T.repeat(3, 1, 1)
print(stacked_inputs.shape)
print('stacked inputs: ', stacked_inputs)
multihead_keys = torch.bmm(multihead_W_key, stacked_inputs)
multihead_values = torch.bmm(multihead_W_value, stacked_inputs)
print("multihead_keys.shape : ", multihead_keys.shape)
print("multihead_values.shape : ", multihead_values.shape)

# Let's swap the second and third dimensions, resulting in tensors with the
# same dimensional structure as the original input sequence, embedded_sentence:

multihead_keys = multihead_keys.permute(0, 2, 1)
multihead_values = multihead_values.permute(0, 2, 1)
print("after swaping the dimension...")
print("multihead_keys.shape : ", multihead_keys.shape)
print("multihead_values.shape : ", multihead_values.shape)

# Let's implement Cross-Attention:
# the only thing that changes is, that we have a 2nd input sequence 
# Suppose we have a sentence with 8 tokens:

embedded_sentence_2 = torch.rand(8, 16) # 2nd input sequence

keys = W_key.matmul(embedded_sentence_2.T).T
values = W_value.matmul(embedded_sentence_2.T).T

print("keys.shape: ", keys.shape)
print("values.shape: ", values.shape)



