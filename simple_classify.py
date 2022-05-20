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
    parser.add_argument('--len_penalty', type=float, default=1.0, help='Length penalty for classification. Default 1, should be 0 for oversmooth regularized models')
    parser.add_argument('--save_prefix', type=str, default='no_save_prefix') 
    parser.add_argument('--split_by_included_funcs', type=str, default=None, help='Annotation csv file. Functions in this file will be found in the main annot_file argument and classified separately from those that aren\'t included. Used to make test P, train F and test P, test F dataset classifications.') 

    args = parser.parse_args()
    print(args)
    return args

def get_train_test_func_inds(train_funcs, test_funcs):
    test_go_term_list = test_funcs.tolist()
    train_funcs = set(train_funcs)
    test_funcs = set(test_funcs)

    test_train_funcs = test_funcs.intersection(train_funcs)
    test_test_funcs = test_funcs.difference(test_train_funcs)

    test_train_func_inds = [test_go_term_list.index(test_train_func) for test_train_func in test_train_funcs]
    test_test_func_inds = [test_go_term_list.index(test_test_func) for test_test_func in test_test_funcs]
    
    return test_train_func_inds, test_test_func_inds


def main(args):
    num_gpus = 1
    model = SeqSet2SeqTransformer.load_from_checkpoint(args.model_checkpoint)
    model.to('cuda:0')
    seq_set_len = args.seq_set_len
    model.len_penalty_param = args.len_penalty

    trainer = Trainer(gpus=num_gpus, logger=False)
    obofile = 'go.obo'
    test_dataset = SequenceGOCSVDataset(args.annot_file, obofile, seq_set_len, vocab=model.vocab)
    if args.split_by_included_funcs is not None:
        train_dataset = SequenceGOCSVDataset(args.split_by_included_funcs, obofile, seq_set_len, vocab=model.vocab)
        test_train_func_inds, test_test_func_inds = get_train_test_func_inds(train_dataset.go_terms, test_dataset.go_terms)
        test_train_func_dataset = SequenceGOCSVDataset(args.annot_file, obofile, seq_set_len, vocab=model.vocab, subset_inds=test_train_func_inds)
        test_test_func_dataset = SequenceGOCSVDataset(args.annot_file, obofile, seq_set_len, vocab=model.vocab, subset_inds=test_test_func_inds)
        with torch.no_grad():
            print('Test prots, train funcs')
            test_train_ground_truth, test_train_preds = classification(model, test_train_func_dataset, save_prefix='classification_pred_files/' + args.save_prefix + '_test_prot_train_functions', num_subsamples=args.num_subsamples)
            print('Test prots, test_funcs')
            test_test_ground_truth, test_test_preds = classification(model, test_test_func_dataset, save_prefix='classification_pred_files/' + args.save_prefix + '_test_prot_test_functions', num_subsamples=args.num_subsamples)
    else:
        with torch.no_grad():
            ground_truth, preds = classification(model, test_dataset, save_prefix='classification_pred_files/' + args.save_prefix, num_subsamples=args.num_subsamples)
    #word_preds = convert_sample_preds_to_words([pred[0] for pred in preds], model.vocab) # get top generation from beam search for each batch

    
if __name__ == '__main__':
    args = get_arguments()
    main(args)
