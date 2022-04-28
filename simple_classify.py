import torch
from data import SequenceGOCSVDataset
from alt_transformer_model import SeqSet2SeqTransformer
from pytorch_lightning import Trainer
import argparse
from train_and_test_model import classification

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('annot_file', type=str, help='Annotation csv file to classify')
    parser.add_argument('model_checkpoint', type=str, help='Model checkpoint used to generate descriptions')
    parser.add_argument('num_subsamples', type=int, help='Number of identically annotated protein sets to subsample')
    parser.add_argument('seq_set_len', type=int, help='Number of proteins per set') 
    parser.add_argument('--save_prefix', type=str, default='no_save_prefix') 

    args = parser.parse_args()
    print(args)
    return args

def main(args):
    num_gpus = 1
    model = SeqSet2SeqTransformer.load_from_checkpoint(args.model_checkpoint)
    model.to('cuda:0')
    seq_set_len = args.seq_set_len

    trainer = Trainer(gpus=num_gpus, logger=False)
    obofile = 'go.obo'
    test_dataset = SequenceGOCSVDataset(args.annot_file, obofile, seq_set_len, vocab=model.vocab)
    with torch.no_grad():
        ground_truth, preds = classification(model, test_dataset, save_prefix='classification_pred_files/' + args.save_prefix, num_subsamples=args.num_subsamples)
    #word_preds = convert_sample_preds_to_words([pred[0] for pred in preds], model.vocab) # get top generation from beam search for each batch

    
if __name__ == '__main__':
    args = get_arguments()
    main(args)
