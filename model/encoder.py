from attention import MultiHeadedAttention
from config import _modelcfg
from torch import nn
from dsconv import DSConv
import torch.nn.functional as F
import logging
logging.basicConfig(level=logging.INFO)



class EmbeddingEncoder(nn.Module):
    """""
    Convert the output of the input embedding
    to an encoded representation. The encoder
    consists of depth-wise convolution x N,
    followed by a multi-head attention layer,
    and followed by a feedforward layer. Each
    processed is followed by layer-normalization
    rather than before.
    """""

    def __init__(self, kernel_size=_modelcfg.kernel_size,
                 input_dim=_modelcfg.input_dim,
                 conv_dim=_modelcfg.conv_dim,
                 heads=_modelcfg.heads,
                 conv_rep=_modelcfg.conv_rep,
                 dropout_p=_modelcfg.dropout_p):
        super().__init__()
        self.mha = MultiHeadedAttention(input_dim=_modelcfg.model_dim, heads=heads)
        self.ff = nn.Linear(in_features=conv_dim, out_features=conv_dim, bias=True)
        self.conv_norm = nn.ModuleList([nn.LayerNorm(conv_dim) for _ in range(conv_rep)])
        self.conv_block = nn.ModuleList([DSConv(kernel_size, conv_dim, conv_dim) for _ in range(conv_rep)])
        self.ln_mha = nn.LayerNorm(_modelcfg.model_dim)
        self.ln_ff = nn.LayerNorm(conv_dim)
        self.dropout_p = dropout_p
        self.pre_conv = DSConv(kernel_size=kernel_size, input_dim=input_dim, out_dim=conv_dim)

    def forward(self, x, mask):
        # print('\nEmbedding Encoder:')
        _x = x
        # print('MHA              ', x_out.size())
        x_out = self.pre_conv(_x)
        x_out = F.dropout(x_out, p=self.dropout_p)
        # print('CONV-RESIZE     ', x_out.size())
        for i, conv in enumerate(self.conv_block):
            res = x_out
            x_out = conv(x_out)
            x_out = F.dropout(x_out, p=self.dropout_p)
            x_out = self.conv_norm[i](x_out + res)
        # print('DSCONV         ', x_out.size())
        x_mha = self.mha(x_out, mask)  # multi-headed attention
        x_mha = F.dropout(x_mha, p=self.dropout_p)
        x_out = self.ln_mha(x_mha + x_out)
        x_ff_temp = x_out
        x = F.dropout(self.ff(x_out), p=self.dropout_p)  # feed-forward
        x = self.ln_ff(x + x_ff_temp)
        # print('EMB-ENC OUT      ', x.size())
        return x

class ModelEncoder(nn.Module):
    """""
    Similar to EmbeddingEncoder
    but differs in encoder layers,
    and convolutions
    """""

    def __init__(self, kernel_size=_modelcfg.kernel_size,
                 input_dim=_modelcfg.model_dim,
                 conv_dim=_modelcfg.conv_dim,
                 heads=_modelcfg.heads,
                 conv_rep=_modelcfg.conv_rep_model,
                 dropout_p=_modelcfg.dropout_p):
        super().__init__()
        self.mha = MultiHeadedAttention(input_dim=input_dim, heads=heads)
        self.ff = nn.Linear(in_features=conv_dim, out_features=conv_dim, bias=True)
        self.conv_norm = nn.ModuleList([nn.LayerNorm(conv_dim) for _ in range(conv_rep)])
        self.conv_block = nn.ModuleList([DSConv(kernel_size, conv_dim, conv_dim) for _ in range(conv_rep)])
        self.ln_mha = nn.LayerNorm(_modelcfg.model_dim)
        self.ln_ff = nn.LayerNorm(conv_dim)
        self.dropout_p = dropout_p

    def forward(self, x, mask):
        x_out = x
        for i, conv in enumerate(self.conv_block):
            res = x_out
            x_out = conv(x_out)
            x_out = F.dropout(x_out, p=self.dropout_p)
            x_out = self.conv_norm[i](x_out + res)

        x_mha = self.mha(x_out, mask)  # multi-headed attention
        x_mha = F.dropout(x_mha, p=self.dropout_p)
        x_out = self.ln_mha(x_mha + x_out)

        x_ff_temp = x_out
        x = F.dropout(self.ff(x_out), p=self.dropout_p)  # feed-forward
        x = self.ln_ff(x + x_ff_temp)
        return x
