import torch
from torch import nn
import torch.nn.functional as F
import math
from config import _modelcfg


def att(q, k, v, mask):
    d_k = q.size(3)
    # query T key
    scaled = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    # apply masking out of padding tokens
    mask = mask.unsqueeze(1).unsqueeze(2)
    scaled = scaled.masked_fill(mask != 0, -1e10)
    # apply softmax
    attention = F.softmax(scaled, dim=-1)
    # multiply qTk by value vector
    values = torch.matmul(attention, v)
    return values, attention


class MultiHeadedAttention(nn.Module):
    def __init__(self, input_dim=None, heads=None):
        super().__init__()
        assert input_dim % heads == 0
        self.input_dim = input_dim
        self.heads = heads
        self.head_dim = input_dim // heads
        self.concat_qkv = nn.Linear(input_dim, 3 * input_dim, bias=False)
        self.out_linear = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, x, mask):
        b, t, k = x.size()
        h = self.heads
        hd = self.head_dim
        assert k == self.input_dim
        # define linear transformation to project sequence to query,key,value vectors
        # apply transformation
        concat_qkv = self.concat_qkv(x)
        # reshape last dimension to number of heads and head dimension * 3
        concat_qkv = concat_qkv.reshape(b, t, h, 3 * hd)
        # swap second and third dimension
        concat_qkv = concat_qkv.permute(0, 2, 1, 3)
        # break tensor by last dim to obtain the separate query,key,value vector
        query, key, value = concat_qkv.chunk(3, dim=-1)
        # apply attention
        values, attention = att(query, key, value, mask=mask)
        # concat all attention head
        values = values.reshape(b, t, h * hd)
        # output vector
        out = self.out_linear(values)
        return out


class QCAttention(nn.Module):
    def __init__(self, model_dim=_modelcfg.model_dim, dropout=_modelcfg.dropout_p):
        """""
        Creates a query aware context vector,
        where the importance of a word in the
        query vector affects the representation
        of words in the context vector but also
        in the opposite. This 
        variation on attention is inspired by
        BIDAF (bi-directional attention-flow).
        """""
        super().__init__()
        self.model_dim = model_dim
        self.dropout = dropout
        self.weight_sim = nn.Linear(3 * model_dim, 1, bias=False)

    def forward(self, C, Q, Cmask, Qmask):

        # line up context and query matrix by adding 300/40 copies along the third dim
        C_sim = C.unsqueeze(2).repeat(1, 1, _modelcfg.question_len, 1)
        Q_sim = Q.unsqueeze(1).repeat(1, _modelcfg.context_len, 1, 1)
        Cmask = Cmask.unsqueeze(2)
        Qmask = Qmask.unsqueeze(1)

        # compute similarity matrix
        CQ_sim = torch.mul(C_sim, Q_sim)

        # concatenate all 3 matrices along third dim
        S = torch.cat([C_sim, Q_sim, CQ_sim], dim=3)
        S = self.weight_sim(S).squeeze()

        # column-wise and row-wise normalization + masking padding tokens
        S1 = F.softmax(S.masked_fill(Qmask != 0, -1e10), dim=2)
        S2 = F.softmax(S.masked_fill(Cmask != 0, -1e10), dim=1)

        # context-to-query
        A = torch.bmm(S1, Q)
        # query-to-context
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)

        # formula identical to the one proposed in the QAnet paper
        # will return a tensor of the same dim as the input except for the last dimension
        out = F.dropout(torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2), p=self.dropout)

        return out






