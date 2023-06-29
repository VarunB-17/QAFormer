import torch
from torch import nn
from config import _modelcfg
import torch.nn.functional as F


class Out(nn.Module):
    def __init__(self):
        """""
        Convert the outputs of the model
        encoder to a vector that consists of 
        start and end probabilities. Implementation
        is identical to the one proposed in the 
        original paper
        """""
        super().__init__()
        self.start_lin = nn.Linear(2 * _modelcfg.model_dim, 1, bias=False)
        self.end_lin = nn.Linear(2 * _modelcfg.model_dim, 1, bias=False)

    def forward(self, M1, M2, M3, Cmask):



        start_cat = torch.cat([M1, M2], dim=2)
        start_cat = self.start_lin(start_cat).squeeze()
        p1 = start_cat.masked_fill(Cmask != 0, -1e10)
        p1 = F.log_softmax(p1, dim=1)

        end_cat = torch.cat([M1, M3], dim=2)
        end_cat = self.end_lin(end_cat).squeeze()
        p2 = end_cat.masked_fill(Cmask != 0, -1e10)
        p2 = F.log_softmax(p2, dim=1)

        return p1, p2
