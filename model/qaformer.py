import torch
from torch import nn
import torch.nn.functional as F
from config import _modelcfg
from input_embedding import Highway, EmbeddingWC
class QaFormer(nn.Module):
    # TODO add embedding layer to convert tensor of tensors of integers to tensors of embeddings
    #       1. For both words and characters -> DONE
    #       2. Apply cnn over character embeddings -> Next STEP
    #       3. Perform max pooling columns wise for character embedding vector
    #       4. Reshape output to fit word-level dimensions
    #       5. Concat word and character embedding matrix
    #       6. Send output to 2-Highway network
    #       7. send Highway output as encoder input

    # TODO add encoder layer that converts question and context embedding to single embedding
    #       input (dim = 500) -> (conv = 128) -> (out = 128  )
    #       1. Apply positional encoding
    #       2. Sent output of pe to repeat convolution block that repeat N times
    #           (k=7, d=128, conv_layers=4)
    #           2.1 Block = layer-norm > conv > add conv output with residual connection
    #           2.2 Output of conv block > layer-norm > attention > attention + residual connection
    #           (heads=8)
    #           2.3 Output of attention > layer-norm > feedforward > feedforward + residual connection
    #       3. Apply step 1 and 2 for both the context and question embedding

    # TODO add context-query attention where context pays attention to the query/question
    #       1.

    def __init__(self, word_emb=torch.load(_modelcfg.word_path), char_emb=torch.load(_modelcfg.char_path)):
        super().__init__()
        self.word_embed = nn.Embedding.from_pretrained(torch.stack(word_emb), freeze=True)
        self.char_embed = nn.Embedding.from_pretrained(torch.stack(char_emb), freeze=False)
        cemb_dim = torch.stack(char_emb).shape[1]
        self.embedding = EmbeddingWC(cemb_dim)

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        print('Raw Input:')
        print(Cwid.size(), Ccid.size(), Qwid.size(), Qcid.size())
        # get word and character embeddings
        Cw, Cc = self.word_embed(Cwid), self.char_embed(Ccid)
        print('\nEmbedded ')
        print(Cw.size(), Cc.size())
        Qw, Qc = self.word_embed(Qwid), self.char_embed(Qcid)
        print(Qw.size(), Qc.size())
        # combine both embeddings by convolution, and concatenation
        C, Q = self.embedding(Cw, Cc), self.embedding(Qw, Qc)
        return C, Q
