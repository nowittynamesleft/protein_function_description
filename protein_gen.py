from transformers import pipeline, GPT2LMHeadModel, AutoTokenizer
import torch
from data import seq_go_collate_pad, SequenceDataset, SequenceGOCSVDataset
from torch.utils.data import DataLoader, Subset
from alt_transformer_model import SeqSet2SeqTransformer
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import numpy as np
import pickle
from functools import partial
from fasta_loader import seq2AAinds
from train_and_test_model import get_prot_preds, convert_preds_to_words, predict_subsample_prots_go_term_descs, classification

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='Assumed to be an annotation csv file with sequences, the same format that is used for training the model.')
    parser.add_argument('model_checkpoint', type=str, help='Model checkpoint used to generate descriptions')
    parser.add_argument('--len_penalty', type=float, default=1.0, help='Length penalty for generation. Default 1, should be 0 for oversmooth regularized models')
    parser.add_argument('--save_prefix', type=str, default='no_save_prefix') 
    args = parser.parse_args()
    print(args)
    return args


def annotation_file_description(annot_file, trainer, model, save_prefix):
    obofile = 'go.obo'
    seq_set_len = 32
    test_dataset = SequenceGOCSVDataset(annot_file, obofile, seq_set_len, vocab=model.vocab)
    test_dl = DataLoader(test_dataset, batch_size=1, collate_fn=partial(seq_go_collate_pad, seq_set_size=seq_set_len), num_workers=1, pin_memory=True)
    predict_subsample_prots_go_term_descs(trainer, model, test_dl, test_dataset, args.save_prefix)


def get_desc_prob(model, dataset, seq_to_gen, save_prefix='no_prefix'):
    GO_padded, GO_pad_masks = dataset.get_padded_descs()
    print(dataset.go_descriptions)
    dataset.set_include_go_mode(False)
    num_subsamples = 1
    ind = 0
    (prot_id_sets, seq_sets) = dataset[ind]
    prot_id_sets = [np.append(prot_id_sets, 'TOGEN')]
    new_seq = seq2AAinds(seq_to_gen)
    seq_sets = [np.array(list(seq_sets) + [new_seq], dtype=object)]
    try:
        S_padded, S_mask = seq_go_collate_pad(list(zip(prot_id_sets, seq_sets)), seq_set_size=len(seq_sets[0])) # batch sizes of 1 each, index out of it
    except TypeError:
        import ipdb; ipdb.set_trace()

    seq_set_desc_probs, seq_set_desc_token_probs = model.classify_seq_sets(S_padded, S_mask, GO_padded, GO_pad_masks) 
    return seq_set_desc_probs


def get_seq_prob(seq):
    tokens = tokenizer.encode(seq, return_tensors='pt').to('cuda:0')
    logits = protgpt2.forward(tokens).logits
    all_probs = torch.softmax(logits, dim=-1)
    seq_probs = torch.gather(all_probs, 2, tokens.unsqueeze(-1))
    log_probs = torch.log(seq_probs)
    sequence_log_prob = torch.sum(log_probs)/len(tokens)**50 # with length penalty on the SEQUENCE 
    return sequence_log_prob


if __name__ == '__main__':
    args = get_arguments()

    protgpt2 = GPT2LMHeadModel.from_pretrained('./ProtGPT2/').to('cuda:0')
    tokenizer = AutoTokenizer.from_pretrained('./ProtGPT2/')

    description = 'Any process that activates or increases the frequency, rate or extent of the non-canonical NF-kappaB cascade.'

    desc_model = SeqSet2SeqTransformer.load_from_checkpoint(args.model_checkpoint)
    desc_model.to('cuda:0')
    desc_model.len_penalty_param = args.len_penalty
    obofile = 'go.obo'
    seq_set_len = 1
    dataset = SequenceGOCSVDataset(args.input_file, obofile, seq_set_len, vocab=desc_model.vocab)
    # replace the dataset's go term with your own
    dataset.tokenize_descriptions([description], desc_model.vocab, 'no_prefix')

    seq = 'MLPGLALLLLAAWTARALEVPTDGNAGLLAEPQIAMFCGRLNMHMNVQNGKWDSDPSGTKTCIDTKEGILQYCQEVYPELQITNVVEANQPVTIQNWCKRGRKQCKTHPHFVIPYRCLVGEFVSDALLVPDKCKFLHQER'
    #CHARS = ['R', 'X', 'S', 'G', 'W', 'I', 'Q', 'A', 'T', 'V', 'K', 'Y', 'C', 'N', 'L', 'F', 'D', 'M', 'P', 'H', 'E', 'U', 'O', 'B', 'Z', '-']
    CHARS = ['R', 'X', 'S', 'G', 'W', 'I', 'Q', 'A', 'T', 'V', 'K', 'Y', 'C', 'N', 'L', 'F', 'D', 'M', 'P', 'H', 'E', 'U', 'O', 'B', 'Z']
    print(len(CHARS))
    print(len(set(seq)))
    max_prob = float('-inf')
    for i in range(100): 
        add_aa = False
        for amino_acid in CHARS:
            possible_seq = seq + amino_acid
            probs = get_desc_prob(desc_model, dataset, possible_seq, save_prefix=args.save_prefix)
            seq_log_prob = get_seq_prob('<|endoftext|>' + seq)
            curr_prob = seq_log_prob.detach().cpu() + probs
            print('Curr seq: ' + possible_seq)
            print('Curr prob: ' + str(curr_prob))
            print('Max prob: ' + str(max_prob))
            if max_prob < curr_prob:
                max_prob = curr_prob
                next_aa = amino_acid
                add_aa = True
        if add_aa:
            seq = seq + next_aa
        else:
            print('Next amino acid does not increase probability. Stopping adding new amino acids.')
            break

            #annotation_file_description(args.input_file, trainer, model, args.save_prefix)


