from data import seq_go_collate_pad, SequenceDataset
from torch.utils.data import DataLoader, Subset
import torch
from alt_transformer_model import SeqSet2SeqTransformer
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import numpy as np
import pickle
from train_and_test_model import get_prot_preds, convert_preds_to_words

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta_file', type=str, help='Fasta file to describe')
    parser.add_argument('model_checkpoint', type=str, help='Model checkpoint used to generate descriptions')
    parser.add_argument('--save_prefix', type=str, default='no_save_prefix') 
    args = parser.parse_args()
    print(args)
    return args

def main(args):
    num_gpus = 1
    model = SeqSet2SeqTransformer.load_from_checkpoint(args.model_checkpoint)
    model.to('cuda:0')
    trainer = Trainer(gpus=num_gpus)
    prot_list, preds, probs = get_prot_preds(args.fasta_file, trainer, model, combined=True)
    #word_preds = convert_sample_preds_to_words([pred[0] for pred in preds], model.vocab) # get top generation from beam search for each batch
    top_preds = [pred[0] for pred in preds]# get top generation from beam search for each batch
    top_probs = [prob[0] for prob in probs]# get top description's probability from beam search for each batch
    word_preds = convert_preds_to_words(top_preds, model.vocab) 
    outfile = open(args.save_prefix + '_generated_descriptions.txt', 'w')
    for i, token_list in enumerate(word_preds):
        gen_desc = ' '.join(token_list)
        outfile.write(gen_desc + '\nProbability: ' + str(torch.exp(top_probs[i]).item()) + '\n\n')
    outfile.close()
    
if __name__ == '__main__':
    args = get_arguments()
    main(args)
