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
from torch.optim.lr_scheduler import CyclicLR

from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)
from analyze_preds import compute_oversmoothing_logratio
from utils import gmean


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        # batch size, 
        pos_embedding = pos_embedding[None, ...]

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
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

    def __init__(self, sigma=1.0):
        super(LengthConverter, self).__init__()
        #self.sigma = nn.Parameter(torch.tensor(1., dtype=torch.float), requires_grad=True)
        self.sigma = torch.tensor(sigma, dtype=torch.float, device=self.device)

    def forward(self, z, tgt_lengths, z_mask):
        #import ipdb; ipdb.set_trace()
        """
        Adjust the number of vectors in `z` according to `tgt_lengths`.
        Return the new `z` and its mask.
        Args:
            z - latent variables, shape: B x L_x x hidden
            tgt_lengths - target lengths, shape: B
            z_mask - latent mask, shape: B x L_x
        """
        curr_lengths = z_mask.sum(1)
        out_pos_inds = torch.arange(tgt_lengths.max().long()).type_as(z)
        in_pos_inds = torch.arange(z.size(1)).type_as(z)

        out_pos_inds = out_pos_inds[None, :].repeat(z.size(0), 1).float() # repeat output positions for whole batch
        len_adjusted_out_pos_inds = out_pos_inds * curr_lengths[:, None].float() / tgt_lengths[:, None].float() # |x|/l_y j
        in_pos_inds = in_pos_inds[None, None, :].repeat(z.size(0), tgt_lengths.max().long(), 1).float()

        # assuming else statement..
        distance = in_pos_inds - len_adjusted_out_pos_inds[:, :, None]
        logits = (- torch.pow(distance, 2) / (2. * self.sigma ** 2))
        #distance = torch.clamp(in_pos_inds - len_adjusted_out_pos_inds[:, :, None], -100, 100)
        #logits = (- torch.pow(2, distance) / (2. * self.sigma ** 2)) # reverse because that's what the old model had I guess

        logits = logits * z_mask[:, None, :].float() - 999. * (1 - z_mask[:, None, :].float()) # remove attention weight from masked indices
        weight = torch.softmax(logits, 2)
        z_prime = torch.matmul(weight, z) # batched matmul
        z_prime_mask = (out_pos_inds < tgt_lengths[:, None].float()).float()
        z_prime = z_prime * z_prime_mask[:, :, None]
        return z_prime, z_prime_mask

    #def compute_monotonic_attention(self, input_positions, output_positions, 
    #        curr_lengths, tgt_lengths):
        


class SeqSet2SeqTransformer(pl.LightningModule):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 emb_size: int, src_vocab_size: int, tgt_vocab_size: int,
                 dim_feedforward:int = 512, num_heads: int = 1, sigma:float = 1.0, dropout:float = 0.1, vocab=None, learning_rate=1e-3,
                 has_scheduler=None, label_smoothing=0.0, oversmooth_param=0.0, len_penalty_param=1.0, pool_op='mean'):
        super(SeqSet2SeqTransformer, self).__init__()
        '''
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=num_heads,
                                                dim_feedforward=dim_feedforward, 
                                                dropout=dropout, batch_first=True)
        '''
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=num_heads,
                                                dim_feedforward=dim_feedforward,
                                                batch_first=True)

        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        '''
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=num_heads, 
                                                dim_feedforward=dim_feedforward, 
                                                dropout=dropout, batch_first=True)
        '''
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=num_heads, 
                                                dim_feedforward=dim_feedforward, 
                                                batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.learning_rate = learning_rate
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.tgt_vocab_size = tgt_vocab_size
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.emb_size = emb_size
        self.len_penalty_param = len_penalty_param
        if label_smoothing != 0.0:
            print('Using label_smoothing ' + str(label_smoothing))
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=0) # ignore padding in calculating loss; 0 is padding index
        self.len_convert = LengthConverter(sigma=sigma)
        self.max_desc_len = 100
        self.vocab = vocab
        self.pred_pair_probs = False
        self.has_scheduler = has_scheduler
        self.oversmooth_param = oversmooth_param
        self.pool_op = pool_op
        if has_scheduler:
            print('Using scheduler, manual optimization.')
            self.automatic_optimization = False
        self.save_hyperparameters()

    def convert_batch_preds_to_words(self, batch):
        word_preds = []
        for sample in batch:
            word_preds.append(self.convert_sample_preds_to_words(sample))
        return word_preds

    def get_probabilities_of_desc_with_no_input(self, tgt, tgt_pad_mask):
        with torch.no_grad():
            src_transformed_list = []
            tgt_in = tgt[:-1].unsqueeze(0)
            tgt_out = tgt[1:]
            #tgt_pad_mask_in = tgt_pad_mask[:-1]
            tgt_pad_mask_out = tgt_pad_mask[1:]
            tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt_in)) # only input for getting description representation for decoder
            tgt_mask = create_target_masks(tgt_in, device=self.device)
            mem_mask = (torch.zeros((1, 1), dtype=bool)).to(self.device)
            
            decoded = self.transformer_decoder(tgt_emb, 
                    torch.zeros(1, 1, self.emb_size).to(self.device), tgt_mask=tgt_mask, memory_key_padding_mask=mem_mask)
            #outputs = self.transformer_decoder(tgt_emb, embedding, 
                    #tgt_mask=tgt_mask, None, GO_pad_mask_input.unsqueeze(0), None) # memory key padding is always assumed to be None, don't need attention mask to hide next description tokens from model
            logits = self.generator(decoded)
            #logits = logits.transpose(1,2)
            #import ipdb; ipdb.set_trace()
            considered_logits = logits[:, ~tgt_pad_mask_out, :]
            considered_probs = torch.softmax(considered_logits, -1)
            considered_log_probs = torch.log(considered_probs)

            correct_tokens = tgt_out[~tgt_pad_mask_out]
            correct_token_log_probs = torch.stack([considered_log_probs[:, position, correct_tokens[position]] for position in range(correct_tokens.shape[0])])
            correct_token_log_probs = correct_token_log_probs.transpose(0,1)
            desc_log_prob = torch.sum(correct_token_log_probs, dim=-1)
            correct_token_log_probs = correct_token_log_probs.detach().cpu().numpy()

        return desc_log_prob
        

    def convert_sample_preds_to_words(self, sample):
        word_preds = [self.vocab[ind] for ind in sample]
        return word_preds


    def _max_lens(self, x_mask):
        ls = torch.max(torch.sum((x_mask == False).float(), dim=1))*torch.ones(x_mask.shape[0], device=self.device) # assuming mask is 0 when sequence should be shown
        return ls


    def forward(self, src: Tensor, trg: Tensor, tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        
        src_transformed_list = []
        avg_src_transformed_embed = self.encode_seq_set(src, src_padding_mask)
        logits = self.decode(trg, avg_src_transformed_embed, tgt_mask)
        logits = logits.transpose(1,2)
        return logits


    def encode_seq_set(self, src, src_padding_mask):
        # want to change to batches
        # Batch size X seq_set_len X max_len (of batch)
        #import ipdb; ipdb.set_trace()
        batch_size, seq_set_len, max_len = src.shape
        transformed_embeds = self.encode(src, src_padding_mask)
        # len convert expects src_padding_mask to be all positions that HAVE tokens, so it is inverted here:
        if self.pool_op == 'gmean':
            transformed_embeds = torch.sigmoid(transformed_embeds)
        transformed_embeds = transformed_embeds.reshape(batch_size*seq_set_len, max_len, -1)
        src_padding_mask = src_padding_mask.reshape(batch_size*seq_set_len, max_len)
        non_padded_max_lens = self._max_lens(src_padding_mask)
        len_trans_embeds, _ = self.len_convert(transformed_embeds, non_padded_max_lens, ~src_padding_mask.bool()) 
        len_trans_embeds = len_trans_embeds.reshape(batch_size, seq_set_len, int(non_padded_max_lens[0].item()), -1)
        if self.pool_op == 'mean':
            avg_src_transformed_embed = torch.mean(len_trans_embeds, dim=1)
        elif self.pool_op == 'gmean':
            avg_src_transformed_embed = gmean(len_trans_embeds, dim=1)
        return avg_src_transformed_embed


    def training_step(self, train_batch, batch_idx):
        S_padded, S_pad_mask, GO_padded_all, GO_pad_mask = train_batch 
        GO_padded_input = GO_padded_all[:, :-1]
        GO_padded_output = GO_padded_all[:, 1:]
        # adding explicit input and output masks for clarity
        GO_pad_mask_input = GO_pad_mask[:, :-1]
        GO_pad_mask_output = GO_pad_mask[:, 1:]
        tgt_mask = create_target_masks(GO_padded_input, device=self.device)

        logits = self(src=S_padded, trg=GO_padded_input, tgt_mask=tgt_mask, src_padding_mask=S_pad_mask,
            tgt_padding_mask=GO_pad_mask_input, memory_key_padding_mask=None)

        eos_idx = self.tgt_vocab_size - 1
        #import ipdb; ipdb.set_trace()
        oversmooth_loss, osr = compute_oversmoothing_logratio(logits.transpose(1,2), GO_padded_output, ~GO_pad_mask_output, eos_idx, margin=1e-5)
        
        _, preds = logits.max(axis=1)
        loss = self.loss_fn(logits, GO_padded_output)
        total_loss = (1 - self.oversmooth_param)*loss + self.oversmooth_param*oversmooth_loss # loss function should ignore padding index (0)
        if batch_idx == 0:
            print('First batch outputs:')
            print(self.convert_batch_preds_to_words(preds))
            print('Actual description:')
            print(self.convert_batch_preds_to_words(GO_padded_output))
            print('Loss for this batch: ' + str(loss))
        self.log("loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("oversmooth_loss", oversmooth_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("sigma", self.len_convert.sigma, on_epoch=True, on_step=False, prog_bar=True)
        self.log("oversmooth_rate", osr, on_epoch=True, on_step=False, prog_bar=True)
        if self.has_scheduler:
            optimizer = self.optimizers()
            self.manual_backward(total_loss) # need to do this when doing manual optimization
            optimizer.step() # need to do this when doing manual optimization
            if self.trainer.is_last_batch: # step scheduler every epoch
                sch = self.lr_schedulers()
                sch.step()
         
        return {'loss': total_loss}
    
    def validation_step(self, valid_batch, batch_idx):
        S_padded, S_pad_mask, GO_padded_all, GO_pad_mask = valid_batch 
        GO_padded_input = GO_padded_all[:, :-1]
        GO_padded_output = GO_padded_all[:, 1:]
        GO_pad_mask_input = GO_pad_mask[:, :-1]
        GO_pad_mask_output = GO_pad_mask[:, 1:]
        tgt_mask = create_target_masks(GO_padded_input, device=self.device)

        outputs = self(src=S_padded, trg=GO_padded_input, tgt_mask=tgt_mask, src_padding_mask=S_pad_mask,
            tgt_padding_mask=GO_pad_mask_input, memory_key_padding_mask=None)

        loss = self.loss_fn(outputs, GO_padded_output)
        loss_dict = {'val_loss': loss}
        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
         
        return loss_dict


    def detect_description_duplicates(self, descriptions):
        for i, description_1 in enumerate(descriptions):
            for j, description_2 in enumerate(descriptions):
                if description_1 == description_2 and i != j:
                    print('DUPLICATED')
                    print(descriptions)
                    return True
        return False


    def get_single_seq_set_desc_pair_logits(self, avg_src_transformed_embeds, actual_GO_padded, tgt_mask, GO_pad_mask):
        # given sequence set embedding and GO description, calculate probability model assigns to the sequence with length penalty
        # making this batchwise based on sequence sets, for a given GO term
        assert 'cuda' in str(self.device)
        with torch.no_grad():
            start_symbol = 0
            end_symbol = self.tgt_vocab_size - 1
            desc_logit = 0
            num_seq_sets = avg_src_transformed_embeds.shape[0]

            actual_GO_padded_input = actual_GO_padded[:, :-1].repeat(num_seq_sets, 1)
            actual_GO_padded_output = actual_GO_padded[0, 1:] # only one GO description
            GO_pad_mask_output = GO_pad_mask[0,1:] # only one mask, function is computing probabilities for multiple seq_sets for one GO description
            
            logits = self.decode(actual_GO_padded_input, avg_src_transformed_embeds, tgt_mask)
            considered_logits = logits[:, ~GO_pad_mask_output, :]
            considered_probs = torch.softmax(considered_logits, -1)
            considered_log_probs = torch.log(considered_probs)

            correct_tokens = actual_GO_padded_output[~GO_pad_mask_output]
            correct_token_log_probs = torch.stack([considered_log_probs[:, position, correct_tokens[position]] for position in range(correct_tokens.shape[0])])
            correct_token_log_probs = correct_token_log_probs.transpose(0,1)
            desc_log_prob = torch.sum(correct_token_log_probs, dim=-1)
            correct_token_log_probs = correct_token_log_probs.detach().cpu().numpy()
             
            desc_log_prob = desc_log_prob/(len(correct_tokens)**self.len_penalty_param) # length penalty; if 0 then no length penalty

        return torch.tensor(desc_log_prob.detach().cpu()), correct_token_log_probs


    def classify_seq_sets(self, S_padded, S_pad_mask, all_GO_padded, GO_pad_mask):
        go_desc_logits = []
        all_desc_token_logits = []
        tgt_mask = create_target_masks(all_GO_padded[:, :-1], device=self.device) # create mask only for input
        embedding = self.encode_seq_set(S_padded.to(self.device), S_pad_mask.to(self.device))
        for go_ind in range(all_GO_padded.shape[0]):
            curr_GO_padded = all_GO_padded[go_ind].to(self.device)
            desc_logits, desc_token_logits = self.get_single_seq_set_desc_pair_logits(embedding, curr_GO_padded.unsqueeze(0), tgt_mask.to(self.device), GO_pad_mask[go_ind].unsqueeze(0).to(self.device))
            go_desc_logits.append(desc_logits)
            all_desc_token_logits.append(desc_token_logits)

        return torch.stack(go_desc_logits).transpose(0,1), all_desc_token_logits


    def beam_search(self, pred_batch):
        print('Beam search')
        print('device:' + str(self.device))
        assert 'cuda' in str(self.device)
        beam_width = 25
        #beam_width = 10
        start_symbol = 0
        end_symbol = self.tgt_vocab_size - 1
        if len(pred_batch) == 4:
            S_padded, S_pad_mask, actual_GO_padded, actual_GO_pad_mask = pred_batch
        elif len(pred_batch) == 2:
            S_padded, S_pad_mask = pred_batch
        num_sets = S_padded.shape[0]
        GO_padded = [torch.ones((1)).fill_(start_symbol).type(torch.long).to(self.device) for i in range(num_sets)] # start GO description off with start token

        all_final_candidate_sentences = []
        all_final_candidate_probs = []

        for seq_set_ind in range(num_sets):
            #import ipdb; ipdb.set_trace()
            embedding = self.encode_seq_set(S_padded[seq_set_ind, ...].unsqueeze(0), S_pad_mask[seq_set_ind, ...].unsqueeze(0))

            curr_GO_padded = GO_padded[seq_set_ind]
            # init candidate sentences
            out, prob = self.get_next_token_probs(curr_GO_padded, embedding)
            prob = torch.softmax(prob.squeeze(), dim=-1)
            candidate_sentences = [curr_GO_padded for i in range(beam_width)] # init with <SOS> token 
            init_top_probs, init_top_words = torch.topk(prob, beam_width, dim=-1, largest=True)

            candidate_probs = torch.log(init_top_probs)
            for beam in range(beam_width):
                candidate_sentences[beam] = torch.cat([candidate_sentences[beam], init_top_words[beam].unsqueeze(0)])

            all_candidates_ended = False
            while not all_candidates_ended:
                sentences = [self.convert_sample_preds_to_words(sample) for sample in candidate_sentences]

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
                        curr_log_probs.append((candidate_probs[beam] + torch.log(curr_top_probs))/(len(candidate_sentences[beam])**self.len_penalty_param))
                        # No len penalty:
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
                    new_candidate_probs.append(unended_top_probs[first_word_beam_ind]*(len(new_candidate_sentence)**self.len_penalty_param))
                    # No length penalty:
                    #new_candidate_probs.append(unended_top_probs[first_word_beam_ind])
                
                # now concatenate the ended candidates and the new candidates, and do a final ranking
                all_probs = torch.cat((torch.Tensor(new_candidate_probs), torch.Tensor(log_probs_ended)))
                all_candidates = new_candidate_sentences + candidates_ended
                top_probs, top_candidate_inds = torch.topk(all_probs.flatten(), beam_width, dim=-1, largest=True)
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
                        candidate_probs = torch.Tensor([candidate_probs[i]/(len(candidate_sentences[i])**self.len_penalty_param) for i in range(len(candidate_probs))])
                        # No len penalty:
                        #candidate_probs = torch.Tensor([candidate_probs[i]) for i in range(len(candidate_probs))])
            candidate_sentences = [sent for _, sent in sorted(zip(candidate_probs, candidate_sentences), key=lambda pair: pair[0], reverse=True)]
            print('Top candidate: ')
            print(' '.join(self.convert_sample_preds_to_words(candidate_sentences[0])))
            all_final_candidate_sentences.append(candidate_sentences)
            all_final_candidate_probs.append(sorted(candidate_probs, reverse=True))
            
            if len(pred_batch) == 4: # if the actual description was supplied in the batch to compare
                print('Actual description:')
                actual_description = ' '.join(self.convert_sample_preds_to_words(actual_GO_padded[seq_set_ind][~actual_GO_pad_mask[seq_set_ind]]))
                print(actual_description)
        return all_final_candidate_sentences, all_final_candidate_probs


    def predict_step(self, pred_batch, batch_idx, dataloader_idx=None):
        if self.pred_pair_probs:
            return self.get_seq_set_desc_pair_probs(pred_batch)
        else:
            return self.beam_search(pred_batch)


    def get_next_token_probs(self, curr_GO_padded, embedding):
        #tgt_mask = (generate_square_subsequent_mask(curr_GO_padded.size(0))).to(self.device) # this is actually unnecessary since this function assumes the whole sequence is known, predicting the next token only
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(curr_GO_padded).unsqueeze(0))
        out = self.transformer_decoder(tgt_emb, embedding) # No masks for this
        out = out.transpose(1, 2)
        prob = self.generator(out[:, :, -1])
        return out, prob


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.has_scheduler: 
            lr_scheduler = CyclicLR(optimizer, cycle_momentum=False, base_lr=self.learning_rate, max_lr=5*self.learning_rate)
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def encode(self, src: Tensor, src_padding_mask: Tensor):
        batch_size, seq_set_len, max_len = src.shape
        src = src.reshape(-1, max_len) # make it one big batch
        src_padding_mask = src_padding_mask.reshape(-1, max_len) # make it one big batch
        encoded = self.transformer_encoder(self.positional_encoding(
                            self.src_tok_emb(src)), None, src_padding_mask)
        encoded = encoded.reshape(batch_size, seq_set_len, max_len, -1) # last dim should be emb_size
        return encoded

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        #import ipdb; ipdb.set_trace()
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt)) # only input for getting description representation for decoder
        decoded = self.transformer_decoder(tgt_emb, memory, tgt_mask)
        #outputs = self.transformer_decoder(tgt_emb, embedding, 
                #tgt_mask=tgt_mask, None, GO_pad_mask_input.unsqueeze(0), None) # memory key padding is always assumed to be None, don't need attention mask to hide next description tokens from model
        logits = self.generator(decoded)

        return logits


def generate_square_subsequent_mask(sz, device=None):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1).to(device)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_target_masks(tgt, device=None):
    tgt_batch_size, tgt_seq_len = tgt.shape
    #tgt_mask = torch.stack([generate_square_subsequent_mask(tgt_seq_len, device=device) for i in range(tgt_batch_size)])
    # tgt_mask should be the same for the whole batch since they're all the same shape if they're padded
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device=device)

    return tgt_mask
