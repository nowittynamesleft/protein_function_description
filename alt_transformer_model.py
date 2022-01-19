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
import numpy as np

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
        n = z_mask.sum(1)
        arange_l = torch.arange(ls.max().long()).type_as(z)
        arange_z = torch.arange(z.size(1)).type_as(z)
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
        logits = (- torch.pow(2, distance) / (2. * self.sigma ** 2))
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
        weight = torch.softmax(logits, 2)
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
                 dim_feedforward:int = 512, num_heads: int = 1, dropout:float = 0.1, vocab=None,
                 min_tf_prob=1.0):
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
        self.tf_prob = 1.0
        self.min_tf_prob = min_tf_prob # default is 1.0, so all teacher forcing; no decrease in ground truth sample probability
        self.pred_pair_probs = False

    def convert_batch_preds_to_words(self, batch):
        word_preds = []
        for sample in batch:
            word_preds.append(self.convert_sample_preds_to_words(sample))
        return word_preds


    def convert_sample_preds_to_words(self, sample):
        word_preds = [self.vocab[ind] for ind in sample]
        return word_preds


    def _max_lens(self, x_mask):
        ls = torch.max(torch.sum((x_mask == False).float(), dim=1))*torch.ones(x_mask.shape[0], device=self.device) # assuming mask is 0 when sequence should be shown
        return ls


    def scheduled_sampling_MM(self, trg, avg_src_transformed_embed, tgt_mask, tgt_padding_mask, teacher_force=1.0):
        # Based on description from Mihaylova and Martins 2019
        # create reference sequence by computing one-by-one the target tokens 
        #import ipdb; ipdb.set_trace()
        with torch.no_grad():
            tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg).unsqueeze(0))
            outs = self.transformer_decoder(tgt_emb, avg_src_transformed_embed, 
                    tgt_mask, None, tgt_padding_mask, None) # memory key padding is always assumed to be None
            predicted_tokens = torch.max(self.generator(outs), dim=-1)[1].squeeze(0)
            pred_tokens_shifted = torch.cat((torch.zeros(1, device=self.device, dtype=torch.long), predicted_tokens[:-1]))
            truth_or_pred = torch.rand(len(trg)).to(self.device) < teacher_force
            reference_desc = torch.where(truth_or_pred, trg, pred_tokens_shifted)

        # second decoding based on reference
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(reference_desc).unsqueeze(0))
        outs = self.transformer_decoder(tgt_emb, avg_src_transformed_embed, 
                tgt_mask, None, tgt_padding_mask, None) # memory key padding is always assumed to be None
        return outs
    
    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor,
                tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        
        src_transformed_list = []
        outputs = []
        #print(src.shape)
        #print('tgt_padding_mask shape')
        #print(tgt_padding_mask.shape) 
        
        for i in range(src.shape[0]): # seq set in batch
            #import ipdb; ipdb.set_trace()

            '''
            src_emb = self.positional_encoding(self.src_tok_emb(src[i, :, :]))
            print(src_emb.shape)
            transformed_embeds = torch.cat([self.transformer_encoder(src_emb[j, :].unsqueeze(0), src_mask[i, j, :], src_padding_mask[i, j, :].unsqueeze(0)) for j in range(src.shape[1])])
            
            len_trans_embeds, _ = self.len_convert(transformed_embeds, self._max_lens(src_padding_mask[i, ...]), src_padding_mask[i, ...].float())

            avg_src_transformed_embed = torch.mean(len_trans_embeds, dim=0).unsqueeze(0)
            '''
            avg_src_transformed_embed = self.encode_seq_set(src[i, ...], src_mask[i, ...], src_padding_mask[i, ...])
            #outs = self.decode(trg[i, :], avg_src_transformed_embed, tgt_mask[i,:])
            #tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg[i, :]).unsqueeze(0))
            #outs = self.transformer_decoder(tgt_emb, avg_src_transformed_embed, 
            #         tgt_mask[i, :], None, tgt_padding_mask[i, :].unsqueeze(0), None) # memory key padding is always assumed to be None
            outs = self.scheduled_sampling_MM(trg[i, :], avg_src_transformed_embed, tgt_mask[i, :], tgt_padding_mask[i, :].unsqueeze(0), teacher_force=self.tf_prob)
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

        #print(outputs.shape)
        _, preds = outputs.max(axis=1)
        #print(self.convert_batch_preds_to_words(preds))
        #print(GO_padded_output.shape)
        loss = self.loss_fn(outputs, GO_padded_output)
        #self.log_dict({'loss': loss, 'sample_output': outputs[0]})
        #self.log_dict({'loss': loss})
        if batch_idx == 0:
            print('First batch first sample outputs:')
            print(self.convert_sample_preds_to_words(preds[0]))
            print('Actual description:')
            print(self.convert_sample_preds_to_words(GO_padded_all[0]))
            print('Teacher forcing prob:')
            print(self.tf_prob)
        self.log("loss", loss)
        if self.tf_prob > self.min_tf_prob:
            self.tf_prob -= 0.000001 # will take 1,000,000 steps to reach 0
         
        return {'loss': loss}
    
    def validation_step(self, valid_batch, batch_idx):
        # get greedy decoding of the first couple samples
        S_padded, S_pad_mask, GO_padded_all, GO_pad_mask = valid_batch 
        GO_padded_input = GO_padded_all[:, :-1]
        GO_padded_output = GO_padded_all[:, 1:]
        GO_pad_mask = GO_pad_mask[:, :-1]
        src_mask, tgt_mask = create_mask(S_padded, GO_padded_input, device=self.device)

        outputs = self(src=S_padded, trg=GO_padded_input, src_mask=src_mask, 
            tgt_mask=tgt_mask, src_padding_mask=S_pad_mask,
            tgt_padding_mask=GO_pad_mask, memory_key_padding_mask=None)

        loss = self.loss_fn(outputs, GO_padded_output)
        loss_dict = {'val_loss': loss}
        #self.log_dict(loss_dict)
        self.log('val_loss', loss)
         
        return loss_dict

    def test_step(self, test_batch, batch_idx):
        S_padded, S_pad_mask, GO_padded, GO_pad_mask = test_batch 
        src_mask, tgt_mask = create_mask(S_padded, GO_padded, device=self.device)

        preds = self.predict_step(test_batch, batch_idx)

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


    def detect_description_duplicates(self, descriptions):
        for i, description_1 in enumerate(descriptions):
            for j, description_2 in enumerate(descriptions):
                if description_1 == description_2 and i != j:
                    print('DUPLICATED')
                    print(descriptions)
                    return True
        return False


    def get_seq_set_desc_pair_probs(self, pred_batch):
        # get correct sequence set GO description pair probabilities
        assert 'cuda' in str(self.device)
        #import ipdb; ipdb.set_trace()
        start_symbol = 0
        end_symbol = self.tgt_vocab_size
        S_padded, S_pad_mask, actual_GO_padded_all, GO_pad_mask = pred_batch
        src_mask, tgt_mask = create_mask(S_padded, actual_GO_padded[:, :-1], device=self.device) # create description mask only for input
        num_sets = S_padded.shape[0]

        desc_log_probs = []
        for seq_set_ind in range(num_sets):
            #print(str(seq_set_ind) + ' out of ' + str(num_sets) + ' sequence sets.')
            curr_actual_GO_padded = actual_GO_padded[seq_set_ind]
            curr_tgt_mask = tgt_mask[seq_set_ind]
            curr_GO_pad_mask = GO_pad_mask[seq_set_ind]

            embedding = self.encode_seq_set(S_padded[seq_set_ind, ...], 
                    src_mask[seq_set_ind, ...], S_pad_mask[seq_set_ind, ...])
            desc_log_prob = self.get_single_seq_set_desc_pair_probs(embedding, curr_actual_GO_padded, curr_tgt_mask, curr_GO_pad_mask)
            #desc_log_prob = self.get_single_seq_set_desc_pair_probs(embedding, curr_actual_GO_padded_input, curr_actual_GO_padded_output, curr_GO_pad_mask_input, curr_GO_pad_mask_output)

            desc_log_probs.append(desc_log_prob)

        desc_log_probs = torch.Tensor(desc_log_probs)

        return desc_log_probs


    def get_single_seq_set_desc_pair_probs(self, embedding, actual_GO_padded, tgt_mask, GO_pad_mask, total_sum=True):
        # given sequence set embedding and GO description, calculate probability model assigns to the sequence with length penalty
        start_symbol = 0
        end_symbol = self.tgt_vocab_size - 1
        desc_log_prob = 0

        #import ipdb; ipdb.set_trace()
        actual_GO_padded_input = actual_GO_padded[:-1]
        actual_GO_padded_output = actual_GO_padded[1:]
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(actual_GO_padded_input).unsqueeze(0)) # only input for getting description representation for decoder
        GO_pad_mask_input = GO_pad_mask[:-1]
        GO_pad_mask_output = GO_pad_mask[1:]

        outputs = self.transformer_decoder(tgt_emb, embedding, 
                tgt_mask, None, GO_pad_mask_input.unsqueeze(0), None) # memory key padding is always assumed to be None, don't need attention mask to hide next description tokens from model
        #outs = self.scheduled_sampling_MM(trg[i, :], avg_src_transformed_embed, tgt_mask[i, :], tgt_padding_mask[i, :].unsqueeze(0), teacher_force=self.tf_prob)
        probs = self.generator(outputs).squeeze()
            
        considered_probs = probs[~GO_pad_mask_output]
        correct_tokens = actual_GO_padded_output[~GO_pad_mask_output]
        correct_token_probs = torch.stack([considered_probs[position, correct_tokens[position]] for position in range(correct_tokens.shape[0])])
        desc_log_prob = torch.sum(correct_token_probs, dim=-1)

        desc_log_prob /= len(correct_tokens) # length penalty, no parameter for now

        if total_sum: 
            return desc_log_prob.item()
        else:
            return correct_token_probs


    def classify_seq_set(self, S_padded, S_pad_mask, all_GO_padded, GO_pad_mask):
        log_probs = []
        src_mask, tgt_mask = create_mask(S_padded, all_GO_padded[:, :-1], device=self.device) # create mask only for input
        #import ipdb; ipdb.set_trace()
        embedding = self.encode_seq_set(S_padded[0].to(self.device), 
                src_mask[0].to(self.device), S_pad_mask[0].to(self.device))
        for go_ind in range(all_GO_padded.shape[0]):
            curr_GO_padded = all_GO_padded[go_ind].to(self.device)
            log_probs.append(self.get_single_seq_set_desc_pair_probs(embedding, curr_GO_padded, tgt_mask[go_ind].to(self.device), GO_pad_mask[go_ind].to(self.device)))

        return log_probs


    def beam_search(self, pred_batch):
        print('Beam search')
        print('device:' + str(self.device))
        assert 'cuda' in str(self.device)
        #import ipdb; ipdb.set_trace()
        beam_width = 25
        start_symbol = 0
        end_symbol = self.tgt_vocab_size - 1
        t = 1.75
        if len(pred_batch) == 4:
            S_padded, S_pad_mask, actual_GO_padded, _ = pred_batch
        elif len(pred_batch) == 2:
            S_padded, S_pad_mask = pred_batch
        num_sets = S_padded.shape[0]
        GO_padded = [torch.ones((1)).fill_(start_symbol).type(torch.long).to(self.device) for i in range(num_sets)] # start GO description off with start token

        src_mask = torch.zeros((S_padded.shape[0], S_padded.shape[1], S_padded.shape[2], S_padded.shape[2]), device=self.device).type(torch.bool)
        all_final_candidate_sentences = []
        all_final_candidate_probs = []

        for seq_set_ind in range(num_sets):
            #import ipdb; ipdb.set_trace()
            embedding = self.encode_seq_set(S_padded[seq_set_ind, ...], src_mask[seq_set_ind, ...], S_pad_mask[seq_set_ind, ...])

            curr_GO_padded = GO_padded[seq_set_ind]
            # init candidate sentences
            out, prob = self.get_next_token_probs(curr_GO_padded, embedding)
            prob = torch.softmax(prob.squeeze(), dim=-1)
            candidate_sentences = [curr_GO_padded for i in range(beam_width)] # init with <SOS> token 
            init_top_probs, init_top_words = torch.topk(prob, beam_width, dim=-1, largest=True)

            candidate_probs = torch.log(init_top_probs)
            for beam in range(beam_width):
                candidate_sentences[beam] = torch.cat([candidate_sentences[beam], init_top_words[beam].unsqueeze(0)])

            #import ipdb; ipdb.set_trace()
            all_candidates_ended = False
            while not all_candidates_ended:
                #print('Current candidates:')
                sentences = [self.convert_sample_preds_to_words(sample) for sample in candidate_sentences]

                #print(sentences)
                curr_log_probs = []
                curr_words = []
                candidates_ended = []
                log_probs_ended = []
                keep_inds = []
                for beam in range(beam_width):
                    if candidate_sentences[beam][-1] != end_symbol:
                        out, prob = self.get_next_token_probs(candidate_sentences[beam], embedding)
                        prob = torch.softmax(prob.squeeze(), dim=-1)
                        curr_top_probs, curr_top_words = torch.topk(prob, beam_width, dim=-1, largest=True)
                        # Length penalty
                        curr_log_probs.append((candidate_probs[beam] + torch.log(curr_top_probs))/(len(candidate_sentences[beam]))**t)
                        #curr_log_probs.append(candidate_probs[beam] + torch.log(curr_top_probs))
                        curr_words.append(curr_top_words)
                        keep_inds.append(beam)
                    else:
                        candidates_ended.append(candidate_sentences[beam])
                        log_probs_ended.append(candidate_probs[beam])
                        
                ended_removed = [candidate_sentences[i] for i in keep_inds]
                candidate_sentences = ended_removed

                next_words = torch.cat(curr_words)
                
                unended_top_probs, unended_top_word_inds = torch.topk(torch.stack(curr_log_probs).flatten(), beam_width, dim=-1, largest=True)
                assert len(next_words) == len(torch.stack(curr_log_probs).flatten())
                #print(self.convert_sample_preds_to_words(next_words))
                # get the actual sentences corresponding to these top log probs
                new_candidate_sentences = []
                new_candidate_probs = []
                for beam in range(beam_width):
                    top_word_ind = unended_top_word_inds[beam]
                    first_word_beam_ind = int(top_word_ind.item()/beam_width)
                    second_word = next_words[top_word_ind]
                    selected_candidate = candidate_sentences[first_word_beam_ind]
                    assert selected_candidate[-1] != end_symbol
                    new_candidate_sentence = torch.cat([selected_candidate, second_word.unsqueeze(0)])
                    new_candidate_sentences.append(new_candidate_sentence)
                    new_candidate_probs.append(unended_top_probs[first_word_beam_ind]*(len(new_candidate_sentence))**t)
                
                # now concatenate the ended candidates and the new candidates, and do a final ranking
                all_probs = torch.cat((torch.Tensor(new_candidate_probs), torch.Tensor(log_probs_ended)))
                all_candidates = new_candidate_sentences + candidates_ended
                top_probs, top_candidate_inds = torch.topk(all_probs.flatten(), beam_width, dim=-1, largest=True)
                #import ipdb; ipdb.set_trace()
                candidate_sentences = (np.array(all_candidates, dtype=object)[top_candidate_inds.numpy()]).tolist()
                candidate_probs = top_probs

                has_duplicates = self.detect_description_duplicates(self.convert_batch_preds_to_words(candidate_sentences))
                if has_duplicates:
                    import ipdb; ipdb.set_trace()
                for sent_ind, candidate_sentence in enumerate(candidate_sentences):
                    if candidate_sentence[-1] != end_symbol and len(candidate_sentence) != self.max_desc_len:
                        break
                    elif sent_ind == len(candidate_sentences) - 1: # if it reached the end without breaking, all candidates end with <EOS>
                        all_candidates_ended = True
                        candidate_probs = torch.Tensor([candidate_probs[i]/len(candidate_sentences[i]) for i in range(len(candidate_probs))])
            #import ipdb; ipdb.set_trace()
            #print(candidate_log_probs)

            #top_prob, top_candidate = torch.max(candidate_probs, 0)
            #GO_padded[seq_set_ind] = candidate_sentences[top_candidate]
            #import ipdb; ipdb.set_trace()
            #print('Top candidate: ')
            #print(' '.join(self.convert_sample_preds_to_words(candidate_sentences[top_candidate])))
            candidate_sentences = [sent for _, sent in sorted(zip(candidate_probs, candidate_sentences), key=lambda pair: pair[0], reverse=True)]
            print('Top candidate: ')
            print(' '.join(self.convert_sample_preds_to_words(candidate_sentences[0])))
            all_final_candidate_sentences.append(candidate_sentences)
            all_final_candidate_probs.append(sorted(candidate_probs, reverse=True))
            
            #print('All candidates:')
            #print(self.convert_batch_preds_to_words(candidate_sentences))
            if len(pred_batch) == 4: # if the actual description was supplied in the batch to compare
                print('Actual description:')
                print(' '.join(self.convert_sample_preds_to_words(actual_GO_padded[seq_set_ind])))
        #return GO_padded
        return all_final_candidate_sentences, all_final_candidate_probs


    def predict_step(self, pred_batch, batch_idx, dataloader_idx=None):
        if self.pred_pair_probs:
            return self.get_seq_set_desc_pair_probs(pred_batch)
        else:
            return self.beam_search(pred_batch)


    def get_next_token_probs(self, curr_GO_padded, embedding):
        tgt_mask = (generate_square_subsequent_mask(curr_GO_padded.size(0))).to(self.device)
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(curr_GO_padded).unsqueeze(0))
        out = self.transformer_decoder(tgt_emb, embedding, 
                None, None, None, None) # memory key padding is always assumed to be None
        out = out.transpose(1, 2)
        prob = self.generator(out[:, :, -1])
        return out, prob

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
    #assert src_batch_size == tgt_batch_size

    tgt_mask = torch.stack([generate_square_subsequent_mask(tgt_seq_len, device=device) for i in range(tgt_batch_size)])
    src_mask = torch.zeros((src_batch_size, src_seq_set_size, src_seq_len, src_seq_len), device=device).type(torch.bool).to(device)

    return src_mask, tgt_mask
