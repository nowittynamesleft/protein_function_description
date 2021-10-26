import math
import torchtext
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive
from torch import Tensor
import io
import time

from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0),:])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class SeqSet2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 emb_size: int, src_vocab_size: int, tgt_vocab_size: int,
                 dim_feedforward:int = 512, num_heads: int = 1, dropout:float = 0.1):
        super(SeqSet2SeqTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=num_heads,
                                                dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=num_heads,
                                                dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor,
                tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        
        src_transformed_list = []
        outputs = []
        print(src.shape)
        print('tgt_padding_mask shape')
        print(tgt_padding_mask.shape) 
        for i in range(src.shape[0]): # seq set in batch
            #import ipdb; ipdb.set_trace()
            src_emb = self.positional_encoding(self.src_tok_emb(src[i, :, :]))
            print(src_emb.shape)
            transformed_embeds = torch.cat([self.transformer_encoder(src_emb[j, :].unsqueeze(0), src_mask[i, j, :], src_padding_mask[i, j, :].unsqueeze(0)) for j in range(src.shape[1])])
            avg_src_transformed_embed = torch.mean(transformed_embeds, dim=0).unsqueeze(0)
            tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg[i, :]).unsqueeze(0))
            outs = self.transformer_decoder(tgt_emb, avg_src_transformed_embed, 
                    tgt_mask[i, :], None, tgt_padding_mask[i, :].unsqueeze(0), None) # memory key padding is always assumed to be None
            outputs.append(self.generator(outs))
            
        return outputs

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer_encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer_decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)


def generate_square_subsequent_mask(sz, device=None):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, device=None):
    # assuming src is a tensor of shape (set_size, max_set_len)
    src_batch_size, src_seq_set_size, src_seq_len = src.shape
    tgt_batch_size, tgt_seq_len = tgt.shape
    assert src_batch_size == tgt_batch_size

    tgt_mask = torch.stack([generate_square_subsequent_mask(tgt_seq_len, device=device) for i in range(tgt_batch_size)])
    src_mask = torch.zeros((src_batch_size, src_seq_set_size, src_seq_len, src_seq_len), device=device).type(torch.bool)

    return src_mask, tgt_mask
