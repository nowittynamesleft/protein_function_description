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
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        #import ipdb; ipdb.set_trace()
        return self.dropout(token_embedding +
                self.pos_embedding[:, :token_embedding.size(1),:])

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
        #import ipdb; ipdb.set_trace()
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
        logits = logits * z_mask[:, None, :].float() - 999. * (1 - z_mask[:, None, :].float())
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
                 dim_feedforward:int = 512, num_heads: int = 1, dropout:float = 0.1, vocab=None):
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
        self.tgt_vocab_size = tgt_vocab_size
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.loss_fn = nn.CrossEntropyLoss()
        self.len_convert = LengthConverter()
        self.max_desc_len = 100
        self.vocab = vocab

    def convert_batch_preds_to_words(self, batch):
        word_preds = []
        for sample in batch:
            word_preds.append([self.vocab[ind] for ind in sample])
        return word_preds


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

            avg_src_transformed_embed = self.encode_seq_set(src[i, ...], src_mask[i, ...], src_padding_mask[i, ...])
            '''
            src_emb = self.positional_encoding(self.src_tok_emb(src[i, :, :]))
            print(src_emb.shape)
            transformed_embeds = torch.cat([self.transformer_encoder(src_emb[j, :].unsqueeze(0), src_mask[i, j, :], src_padding_mask[i, j, :].unsqueeze(0)) for j in range(src.shape[1])])
            
            len_trans_embeds, _ = self.len_convert(transformed_embeds, self._max_lens(src_padding_mask[i, ...]), src_padding_mask[i, ...].float())

            avg_src_transformed_embed = torch.mean(len_trans_embeds, dim=0).unsqueeze(0)
            '''
            #outs = self.decode(trg[i, :], avg_src_transformed_embed, tgt_mask[i,:])
            tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg[i, :]).unsqueeze(0))
            outs = self.transformer_decoder(tgt_emb, avg_src_transformed_embed, 
                    tgt_mask[i, :], None, tgt_padding_mask[i, :].unsqueeze(0), None) # memory key padding is always assumed to be None
            outputs.append(self.generator(outs))
            
        outputs = torch.stack(outputs).squeeze(1)
        outputs = outputs.transpose(1,2)
        return outputs

    def encode_seq_set(self, src, src_mask, src_padding_mask):
        #import ipdb; ipdb.set_trace()
        transformed_embeds = torch.cat([self.encode(src[j, :].unsqueeze(0), src_mask[j, :], src_padding_mask[j, :].unsqueeze(0)) for j in range(src.shape[0])])
        len_trans_embeds, _ = self.len_convert(transformed_embeds, self._max_lens(src_padding_mask), ~src_padding_mask.bool()) # len convert seems to have strange behavior (all features at a given position are the same after it)
        # len convert expects src_padding_mask to be all positions that HAVE tokens, so it is inverted here.
        avg_src_transformed_embed = torch.mean(len_trans_embeds, dim=0).unsqueeze(0)
        return avg_src_transformed_embed

    def training_step(self, train_batch, batch_idx):
        S_padded, S_pad_mask, GO_padded_all, GO_pad_mask = train_batch 
        GO_padded_input = GO_padded_all[:, :-1]
        GO_padded_output = GO_padded_all[:, 1:]
        GO_pad_mask = GO_pad_mask[:, :-1]
        src_mask, tgt_mask = create_mask(S_padded, GO_padded_input, device=self.device)

        outputs = self(src=S_padded, trg=GO_padded_input, src_mask=src_mask, 
            tgt_mask=tgt_mask, src_padding_mask=S_pad_mask,
            tgt_padding_mask=GO_pad_mask, memory_key_padding_mask=None)

        print(outputs.shape)
        _, preds = outputs.max(axis=1)
        print(self.convert_batch_preds_to_words(preds))
        print(GO_padded_output.shape)
        loss = self.loss_fn(outputs, GO_padded_output)
        self.log_dict({'loss': loss})
        #self.log("loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
         
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
         
        return {'val_loss': loss}

    def test_step(self, test_batch, batch_idx):
        S_padded, S_pad_mask, GO_padded, GO_pad_mask = test_batch 
        src_mask, tgt_mask = create_mask(S_padded, GO_padded, device=self.device)

        preds = self.predict_step(test_batch, batch_idx)

        print(GO_padded.shape)
        #_, preds = outputs.max(axis=1)
        acc = 0
        #import ipdb; ipdb.set_trace()
        end_symbol = self.tgt_vocab_size - 1
        for i in range(len(preds)):
            curr_GO_padded = GO_padded[i]
            stripped_ind_list = []
            for val in curr_GO_padded:
                symbol = val.item()
                stripped_ind_list.append(symbol)
                if symbol == end_symbol:
                    break
            # stripped_ind_list now has only tokens that are before and including <EOS>
            try:
                if preds[i].tolist() == stripped_ind_list:
                    acc += 1
            except RuntimeError:
                pass
        #loss = self.loss_fn(outputs, GO_padded)
        acc /= len(preds)
        self.log_dict({'acc': acc})

    def predict_step(self, pred_batch, batch_idx):
        '''
        TODO: make greedy decode step instead to actually make this a prediction method without requiring the expected output
        '''
        start_symbol = 0 # is this the index of the start token? prob not
        end_symbol = self.tgt_vocab_size - 1 # is this the index of the end token?
        if len(pred_batch) == 4:
            S_padded, S_pad_mask, _, _ = pred_batch
        num_sets = S_padded.shape[0]
        GO_padded = [torch.ones((1)).fill_(start_symbol).type(torch.long).to(self.device) for i in range(num_sets)] # start GO description off with start token

        #src_mask, tgt_mask = create_mask(S_padded, GO_padded, device=self.device)
        src_mask = torch.zeros((S_padded.shape[0], S_padded.shape[1], S_padded.shape[2], S_padded.shape[2]), device=self.device).type(torch.bool).to(self.device)

        for seq_set_ind in range(num_sets):
            #import ipdb; ipdb.set_trace()
            embedding = self.encode_seq_set(S_padded[seq_set_ind, ...], src_mask[seq_set_ind, ...], S_pad_mask[seq_set_ind, ...])
            #embedding = embedding.to(self.device)

            curr_GO_padded = GO_padded[seq_set_ind]
            #ended_desc_mask = torch.zeros(num_sets, dtype=bool)
            for i in range(self.max_desc_len - 1): # trying to make this batch wise but how do you break the loop for those that have an end symbol produced?
                #import ipdb; ipdb.set_trace()
                #memory_mask = torch.zeros(GO_padded.shape[0], memory.shape[0]).to(self.device).type(torch.bool)
                
                #import ipdb; ipdb.set_trace()
                tgt_mask = (generate_square_subsequent_mask(curr_GO_padded.size(0))).to(self.device)
                #out = self.decode(GO_padded[seq_set_ind,:].unsqueeze(0), embedding, tgt_mask)

                tgt_emb = self.positional_encoding(self.tgt_tok_emb(curr_GO_padded).unsqueeze(0))
                #out = self.transformer_decoder(tgt_emb, embedding, 
                #        tgt_mask, None, None, None) # memory key padding is always assumed to be None
                out = self.transformer_decoder(tgt_emb, embedding, 
                        None, None, None, None) # memory key padding is always assumed to be None
                out = out.transpose(1, 2)
                prob = self.generator(out[:, :, -1])
                #prob[..., 0] = 0
                _, next_word = torch.max(prob, dim = -1)
                curr_GO_padded = torch.cat((curr_GO_padded, next_word), dim=0)
                next_word = next_word.item()
                '''
                _, batch_next_words = torch.max(prob, dim = -1)
                for j, word in enumerate(batch_next_words): # if any produced an end token in the past, permanently make all tokens afterwards end until every set is done
                    if word == end_symbol:
                        ended_desc_mask[j] = True
                batch_next_words[ended_desc_mask] = end_symbol
                GO_padded = torch.cat(GO_padded, batch_next_words, dim=1)
                if torch.all(batch_next_words == end_symbol):
                  break
                '''

                if next_word == end_symbol:
                    break
            GO_padded[seq_set_ind] = curr_GO_padded

        return GO_padded

    '''
    def greedy_decode(model, src, src_mask, max_len, start_symbol):
        src = src.to(device)
        src_mask = src_mask.to(device)

        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
        for i in range(max_len-1):
            memory = memory.to(device)
            memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                                        .type(torch.bool)).to(device)
            out = model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim = 1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == EOS_IDX:
              break
        return ys
    '''

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def encode(self, src: Tensor, src_mask: Tensor, src_padding_mask: Tensor):
        #import ipdb; ipdb.set_trace()
        return self.transformer_encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask, src_padding_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        #import ipdb; ipdb.set_trace()
        return self.transformer_decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory, tgt_mask)


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
