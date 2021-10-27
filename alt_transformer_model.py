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
import pytorch_lightning as pl

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


class LengthConverter(pl.LightningModule):
    """
    Implementation of Length Transformation. From Shu et al 2019
    """

    def __init__(self):
        super(LengthConverter, self).__init__()
        self.sigma = nn.Parameter(torch.tensor(1., dtype=torch.float), requires_grad=True)

    def forward(self, z, ls, z_mask):
        """
        Adjust the number of vectors in `z` according to `ls`.
        Return the new `z` and its mask.
        Args:
            z - latent variables, shape: B x L_x x hidden
            ls - target lengths, shape: B
            z_mask - latent mask, shape: B x L_x
        """
        n = z_mask.sum(1).to(self.device)
        arange_l = torch.arange(ls.max().long()).to(self.device)
        arange_z = torch.arange(z.size(1)).to(self.device)
        '''
        if torch.cuda.is_available():
            arange_l = arange_l.cuda()
            arange_z = arange_z.cuda()
        '''
        arange_l = arange_l[None, :].repeat(z.size(0), 1).float()
        mu = arange_l * n[:, None].float() / ls[:, None].float()
        arange_z = arange_z[None, None, :].repeat(z.size(0), ls.max().long(), 1).float()

        # assuming else statement..
        distance = torch.clamp(arange_z - mu[:, :, None], -100, 100)
        logits = (- torch.pow(2, distance) / (2. * self.sigma ** 2)).to(self.device)
        '''
        if OPTS.fp16:
            arange_l = arange_l.half()
            mu = mu.half()
            arange_z = arange_z.half()
        if OPTS.fixbug1:
            logits = - torch.pow(arange_z - mu[:, :, None], 2) / (2. * self.sigma ** 2)
        else:
            distance = torch.clamp(arange_z - mu[:, :, None], -100, 100)
            logits = - torch.pow(2, distance) / (2. * self.sigma ** 2)
        '''
        logits = logits * z_mask[:, None, :] - 999. * (1 - z_mask[:, None, :])
        weight = torch.softmax(logits, 2).to(self.device)
        # z_prime = (z[:, None, :, :] * weight[:, :, :, None]).sum(2) # use torch.multiply instead, to do dot product between two matrices
        z_prime = torch.matmul(weight, z)
        """
        if OPTS.fp16:
            z_prime_mask = (arange_l < ls[:, None].half()).half()
        else:
            z_prime_mask = (arange_l < ls[:, None].float()).float()
        """
        z_prime_mask = (arange_l < ls[:, None].float()).float()
        z_prime = z_prime * z_prime_mask[:, :, None]
        return z_prime, z_prime_mask


class SeqSet2SeqTransformer(pl.LightningModule):
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
        self.loss_fn = nn.CrossEntropyLoss()
        self.len_convert = LengthConverter()

    def _max_lens(self, x_mask):
        ls = torch.max(torch.sum((x_mask == False).float(), dim=1))*torch.ones(x_mask.shape[0], device=self.device) # assuming mask is 0 when sequence should be shown
        return ls

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

            len_trans_embeds, _ = self.len_convert(transformed_embeds, self._max_lens(src_padding_mask[i, ...]), src_padding_mask[i, ...].float())

            avg_src_transformed_embed = torch.mean(len_trans_embeds, dim=0).unsqueeze(0)
            tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg[i, :]).unsqueeze(0))
            outs = self.transformer_decoder(tgt_emb, avg_src_transformed_embed, 
                    tgt_mask[i, :], None, tgt_padding_mask[i, :].unsqueeze(0), None) # memory key padding is always assumed to be None
            outputs.append(self.generator(outs))
            
        outputs = torch.stack(outputs).squeeze(1)
        outputs = outputs.transpose(1,2)
        return outputs

    def training_step(self, train_batch, batch_idx):
        S_padded, S_pad_mask, GO_padded, GO_pad_mask = train_batch 
        src_mask, tgt_mask = create_mask(S_padded, GO_padded, device=self.device)

        outputs = self(src=S_padded, trg=GO_padded, src_mask=src_mask, 
            tgt_mask=tgt_mask, src_padding_mask=S_pad_mask,
            tgt_padding_mask=GO_pad_mask, memory_key_padding_mask=None)

        print(outputs.shape)
        print(GO_padded.shape)
        loss = self.loss_fn(outputs, GO_padded)
         
        return {'loss': loss}


    def validation_step(self, valid_batch, batch_idx):
        S_padded, S_pad_mask, GO_padded, GO_pad_mask = valid_batch 
        src_mask, tgt_mask = create_mask(S_padded, GO_padded, device=self.device)

        outputs = self(src=S_padded, trg=GO_padded, src_mask=src_mask, 
            tgt_mask=tgt_mask, src_padding_mask=S_pad_mask,
            tgt_padding_mask=GO_pad_mask, memory_key_padding_mask=None)

        print(outputs.shape)
        print(GO_padded.shape)
        loss = self.loss_fn(outputs, GO_padded)
         
        return {'loss': loss}


    def predict_step(self, pred_batch, batch_idx):
        S_padded, S_pad_mask, GO_padded, GO_pad_mask = pred_batch 
        src_mask, tgt_mask = create_mask(S_padded, GO_padded, device=self.device)

        outputs = self(src=S_padded, trg=GO_padded, src_mask=src_mask, 
            tgt_mask=tgt_mask, src_padding_mask=S_pad_mask,
            tgt_padding_mask=GO_pad_mask, memory_key_padding_mask=None)

        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer_encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer_decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)


def generate_square_subsequent_mask(sz, device=None):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1).to(device)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, device=None):
    # assuming src is a tensor of shape (set_size, max_set_len)
    src_batch_size, src_seq_set_size, src_seq_len = src.shape
    tgt_batch_size, tgt_seq_len = tgt.shape
    assert src_batch_size == tgt_batch_size

    tgt_mask = torch.stack([generate_square_subsequent_mask(tgt_seq_len, device=device) for i in range(tgt_batch_size)])
    src_mask = torch.zeros((src_batch_size, src_seq_set_size, src_seq_len, src_seq_len), device=device).type(torch.bool).to(device)

    return src_mask, tgt_mask
