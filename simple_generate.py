from data import seq_go_collate_pad, SequenceDataset, SequenceGOCSVDataset
from torch.utils.data import DataLoader, Subset
import torch
from alt_transformer_model import SeqSet2SeqTransformer
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import numpy as np
import pickle
from functools import partial
from train_and_test_model import get_prot_preds, convert_preds_to_words, predict_subsample_prots_go_term_descs

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='Fasta file to describe; if --annot_file flag is used it is assumed to be an annotation csv file with sequences, the same format that is used for training the model.')
    parser.add_argument('model_checkpoint', type=str, help='Model checkpoint used to generate descriptions')
    parser.add_argument('--save_prefix', type=str, default='no_save_prefix') 
    parser.add_argument('--annot_file', action='store_true', 
            help='Changes behavior to open the first argument')
    args = parser.parse_args()
    print(args)
    return args

def fasta_description(fasta_file, trainer, model, save_prefix):
    prot_list, preds, probs = get_prot_preds(fasta_file, trainer, model, combined=True)
    top_preds = [pred[0] for pred in preds]# get top generation from beam search for each batch
    top_probs = [prob[0] for prob in probs]# get top description's probability from beam search for each batch
    word_preds = convert_preds_to_words(top_preds, model.vocab) 
    outfile = open(args.save_prefix + '_generated_descriptions.txt', 'w')
    for i, token_list in enumerate(word_preds):
        gen_desc = ' '.join(token_list)
        outfile.write(gen_desc + '\nProbability: ' + str(torch.exp(top_probs[i]).item()) + '\n\n')
    outfile.close()

def annotation_file_description(annot_file, trainer, model, save_prefix):
    obofile = 'go.obo'
    seq_set_len = 32
    test_dataset = SequenceGOCSVDataset(annot_file, obofile, seq_set_len, vocab=model.vocab)
    test_dl = DataLoader(test_dataset, batch_size=1, collate_fn=partial(seq_go_collate_pad, seq_set_size=seq_set_len), num_workers=1, pin_memory=True)
    predict_subsample_prots_go_term_descs(trainer, model, test_dl, test_dataset, args.save_prefix)
    

def main(args):
    num_gpus = 1
    model = SeqSet2SeqTransformer.load_from_checkpoint(args.model_checkpoint)
    model.to('cuda:0')
    trainer = Trainer(gpus=num_gpus)
    if not args.annot_file:
        fasta_description(args.input_file, trainer, model, args.save_prefix)
    else:
        annotation_file_description(args.input_file, trainer, model, args.save_prefix)
    
    
if __name__ == '__main__':
    args = get_arguments()
    main(args)
