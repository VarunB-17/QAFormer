import torch
from torch import nn
import torch.nn.functional as F
from config import _modelcfg
from input_embedding import EmbeddingWC
from encoder import EmbeddingEncoder, ModelEncoder
from attention import QCAttention
from dsconv import DSConv
from span import Out


class QaFormer(nn.Module):
    def __init__(self, word_emb=torch.load(_modelcfg.word_path), char_emb=torch.load(_modelcfg.char_path)):
        super().__init__()
        self.word_embed = nn.Embedding.from_pretrained(torch.stack(word_emb), freeze=True)
        self.char_embed = nn.Embedding.from_pretrained(torch.stack(char_emb), freeze=False)
        cemb_dim = torch.stack(char_emb).shape[1]
        self.embedding = EmbeddingWC(cemb_dim)
        self.c_encoder = EmbeddingEncoder()
        self.q_encoder = EmbeddingEncoder()
        self.qca = QCAttention()
        self.qca_resizer = DSConv(kernel_size=_modelcfg.kernel_size,
                                  input_dim=_modelcfg.model_dim * 4,
                                  out_dim=_modelcfg.model_dim)
        self.model_enc = nn.ModuleList([ModelEncoder() for _ in range(_modelcfg.enc_layer)])
        self.out = Out()

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        Cmask = torch.eq(Cwid, 0).float()
        Qmask = torch.eq(Qwid, 0).float()

        # get word and character embeddings
        Cw, Cc = self.word_embed(Cwid), self.char_embed(Ccid)

        Qw, Qc = self.word_embed(Qwid), self.char_embed(Qcid)

        # combine both embeddings by convolution, and concatenation
        C, Q = self.embedding(Cw, Cc), self.embedding(Qw, Qc)
        C_enc, Q_enc = self.c_encoder(C, Cmask), self.q_encoder(Q, Qmask)

        CQatt = self.qca(C_enc, Q_enc, Cmask, Qmask)

        M1 = self.qca_resizer(CQatt)
        M1 = F.dropout(M1, p=_modelcfg.dropout_p)

        for i, module in enumerate(self.model_enc):
            M1 = module(M1, Cmask)

        M2 = M1

        M2 = F.dropout(M2, p=_modelcfg.dropout_p)
        for i, module in enumerate(self.model_enc):
            M2 = module(M2, Cmask)

        M3 = M2

        M3 = F.dropout(M3, p=_modelcfg.dropout_p)
        for i, module in enumerate(self.model_enc):
            M3 = module(M1, Cmask)

        start, end = self.out(M1, M2, M3, Cmask)

        return start, end
