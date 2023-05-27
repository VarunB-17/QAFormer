import torch
from config import _modelcfg
from torch import nn
import torch.nn.functional as F
import logging


class Highway(nn.Module):
    """""
    allows for better flow of information across deep models and take into account
    the vanishing gradient problem by learning weight matrices that determine
    what information if passed and what not. Code inspired by ->
    https://github.com/hengruo/QANet-pytorch/blob/master/models.py
    """""

    def __init__(self, layers=_modelcfg.highway_layer, size=_modelcfg.input_dim):
        super().__init__()
        self.layers = layers
        self.transform = nn.ModuleList([nn.Linear(size, size) for _ in range(self.layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(self.layers)])

    def forward(self, x):

        for i in range(self.layers):
            transform = F.relu(self.transform[i](x))
            gate = F.sigmoid((self.gate[i](x)))
            x = gate * transform + (1 - gate) * x

        return x


class EmbeddingWC(nn.Module):
    """""
    Converts a id representation of a batch of sequences
    to their corresponding embedding vectors. This is applied
    to both word-level and character-level based embeddings,
    where word embedding are obtained be pre-trained embedding,
    and, where character embedding is set to be random and is passed
    through a convolution to learn local features.
    Both embeddings get concatenated after processing, so that
    each word is represented by itself and its characters.
    """""

    def __init__(self, c_dim=_modelcfg.char_emb):
        super().__init__()
        self.conv_char = nn.Conv2d(in_channels=c_dim,
                                   out_channels=c_dim,
                                   kernel_size=(1, 5),
                                   padding=(0, 2),
                                   bias=True)
        self.highway = Highway()

    def forward(self, word_emb, char_emb):
        print('\nConvolution:')
        char_emb = char_emb.permute(0, 3, 1, 2)
        char_emb = F.dropout(char_emb, p=_modelcfg.c_drop, training=self.training)
        char_emb = self.conv_char(char_emb)
        print('after conv           ', char_emb.size())
        char_emb = F.relu(char_emb)
        char_emb = char_emb.permute(0, 2, 3, 1)
        print('after reshape            ', char_emb.size())
        char_emb = torch.max(char_emb, dim=2)[0]
        print('after  max           ', char_emb.size())

        word_emb = F.dropout(word_emb, p=_modelcfg.w_drop, training=self.training)
        emb = torch.cat([char_emb, word_emb], dim=2)
        print('concat word+char         ', emb.size())
        emb = self.highway(emb)
        print('highway out:             ', emb.size())
        return None
