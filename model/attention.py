import torch
from torch import nn
import torch.nn.functional as F
import math
from batch_functions import get_device
device = get_device()




def att(q, k, v, mask=False):
    d_k = q.size(3)
    # query T key
    scaled = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    # apply masking
    if mask:
        mask = torch.full(scaled.size(), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        scaled += mask
    # apply softmax
    attention = F.softmax(scaled, dim=-1)
    # multiply qTk by value vector
    values = torch.matmul(attention, v)
    return values, attention


class MultiHeadedAttention(nn.Module):
    def __init__(self, input_dim, heads=4, mask=False):
        super().__init__()
        assert input_dim % heads == 0
        self.input_dim = input_dim
        self.heads = heads
        self.head_dim = input_dim // heads
        self.mask = mask
        self.concat_qkv = nn.Linear(input_dim, 3 * input_dim, bias=False)
        self.out_linear = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads
        hd = self.head_dim
        assert k == self.input_dim
        # define linear transformation to project sequence to query,key,value vectors
        # apply transformation
        concat_qkv = self.concat_qkv(x)  # torch.Size([190, 43, 1536])
        # reshape last dimension to number of heads and head dimension * 3
        concat_qkv = concat_qkv.reshape(b, t, h, 3 * hd)  # torch.Size([190, 43, 4, 384])
        # swap second and third dimension
        concat_qkv = concat_qkv.permute(0, 2, 1, 3)  # torch.Size([190, 4, 43, 384])
        # break tensor by last dim to obtain the separate query,key,value vector
        query, key, value = concat_qkv.chunk(3, dim=-1)
        # apply attention
        values, attention = att(query, key, value, mask=self.mask)
        # concat all attention head
        values = values.reshape(b, t, h * hd)
        # output vector
        out = self.out_linear(values)
        return out
