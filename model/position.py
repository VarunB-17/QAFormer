import torch
from torch import nn
from dev import get_device

device = get_device()

class PositionalEncoding(nn.Module):
    """""
    Addition of a vector to
    a word embedding based on
    its relative positioning
    using the sine and cosine
    functions and the dimension
    of an embedding vector.
    """""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        # tensor of length embedding dimension
        pos_even = torch.arange(0, self.embed_dim, 2, dtype=torch.float32)
        #  compute denominator
        denominator = torch.pow(10000, pos_even / self.embed_dim)
        #  define empty positional embedding
        pos = torch.arange(x.size(1), dtype=torch.float32).unsqueeze(1)
        #  compute embedding values for odd and even positional
        even = torch.sin(pos / denominator)
        odd = torch.cos(pos / denominator)
        #  interleave odd and even matrices among second dimension
        pos_embed = torch.stack([even, odd], dim=2)
        pos_embed = torch.flatten(pos_embed, start_dim=1, end_dim=2)
        # broadcast along shape of batch size in order to match dimensions of input seq
        pos_embed = pos_embed.expand(x.size(0), -1, -1)
        #  add positional embedding to word embedding
        embedding = x.to(device) + pos_embed.to(device)

        return embedding
