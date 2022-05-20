import sys
import numpy as np
from utils import get_last_epoch_checkpoint, ensure_dir
import pickle
import argparse


'''
Generates a set of scripts for first training the models with different hyperparameter settings,
then using the last checkpoints of those models to classify, and generate descriptions for a set of validation GO terms
'''

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('type_of_run', type=str, help='train, oversmooth_tune, classify or generate')
    parser.add_argument('prefix', type=str)
    parser.add_argument('--cutoff_dataset', action="store_true", help='Use the cutoff dataset instead of the full dataset.')
    parser.add_argument('--bpe_dataset', action="store_true", help='Use the no-cutoff bpe dataset instead of the regular word dataset.')
    parser.add_argument('--num_hparam_sets', type=int, default=None, help='Required only when train type of run; samples hparams from dict to run.')
    parser.add_argument('--no_len_penalty', action="store_true", help='Remove length penalty for generation/classification.')
    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    args = arguments()
    type_of_run = args.type_of_run
    prefix = args.prefix
    if type_of_run == 'oversmooth_tune':
        prefix += '_oversmooth_param_0.9'
    hyperparam_list_fname = prefix + '_hparams.pckl'
    if type_of_run == 'train':
        num_hparam_sets = args.num_hparam_sets
    elif args.num_hparam_sets is not None:
        print('There should not be a number of sampled hparam sets for classify or generate settings. There should be a hyperparam list pickle that is loaded named: ' + hyperparam_list_fname)
        print('Exiting without making script.')
        exit()
    if args.cutoff_dataset and not args.bpe_dataset:
        train_set = 'uniprot_sprot_training_training_split.csv'
        val_set = 'uniprot_sprot_training_val_split.csv'
        classify_gen_set = val_set
    elif args.bpe_dataset:
        train_set = 'uniprot_sprot_annot_no_go_cutoff_randomly_split_train_set_codified_1000_ops.tsv'
        val_set = 'uniprot_sprot_annot_no_go_cutoff_randomly_split_validation_set_codified_1000_ops.tsv'
        classify_gen_set = 'random_split_first_1000_val_set_codified_1000_ops.tsv'
    else:
        train_set = 'uniprot_sprot_annot_no_go_cutoff_randomly_split_train_set.csv'
        val_set = 'uniprot_sprot_annot_no_go_cutoff_randomly_split_validation_set.csv'
        classify_gen_set = 'random_split_first_1000_val_set.csv'
    # Architecture search hyperparameter settings:
    hyperparam_dict = { 'num_encoder_layers': [1, 2],
                        'num_decoder_layers': [1, 2],
                        'dim_feedforward': [128, 256, 512],
                        'emb_size': [128, 256],
                        'sigma': [0.25, 0.5, 1.0, 2.0],
                        'dropout': [0.0, 0.25],
                        'seq_set_len': [32],
                        'learning_rate': [1e-4, 5e-4, 1e-3, 2e-3],
                        'label_smoothing': [0.0, 0.1, 0.2],
                        #'oversmooth_param': [0.0, 0.1, 0.2] # just tune after training
                        }
    '''
    hyperparam_dict = { 'num_encoder_layers': [1],
                        'num_decoder_layers': [1],
                        'dim_feedforward': [512],
                        'emb_size': [256],
                        'sigma': [0.5],
                        'dropout': [0.0],
                        'seq_set_len': [32],
                        'learning_rate': [1e-3]
                        }
                        #full_data_architecture_search_num_encoder_layers-1_num_decoder_layers-1_emb_size-256_sigma-0.5_seq_set_len-32_experiment
    '''
    if type_of_run == 'train':
        hyperparam_sets = []
        for i in range(num_hparam_sets):
            new_set = {}
            for param in hyperparam_dict.keys():
                selected_setting = np.random.choice(hyperparam_dict[param])
                new_set[param] = selected_setting
            hyperparam_sets.append(new_set)
        pickle.dump(hyperparam_sets, open(hyperparam_list_fname, 'wb'))
    else:
        hyperparam_sets = pickle.load(open(hyperparam_list_fname, 'rb'))
         
    num_subsamples = 4
    classify_seq_set_len = 32
    outfile = open(prefix + '_' + type_of_run + '.sh', 'w')
    ensure_dir('run_log_files/')
    if type_of_run == 'train' or type_of_run == 'oversmooth_tune':
        first_part = 'sbatch -N 1 -p gpu --gres=gpu:1 --mem 150G --wrap \"python train_and_test_model.py ' + train_set + ' ' + val_set + ' '
    elif type_of_run == 'classify':
        first_part = 'sbatch -N 1 -p gpu --gres=gpu:1 --mem 150G --wrap \"python simple_classify.py ' + classify_gen_set + ' '
    elif type_of_run == 'generate':
        first_part = 'sbatch -N 1 -p gpu --gres=gpu:1 --mem 150G --wrap \"python simple_generate.py ' + classify_gen_set + ' '
    for hyperparam_set in hyperparam_sets:
        strings = ['--' + key + ' ' + str(value) for (key, value) in hyperparam_set.items()]
        save_prefix = prefix + '_' + '_'.join([key + '-' + str(value) for (key, value) in hyperparam_set.items()])
        experiment_folder = 'lightning_logs/' + save_prefix + '_experiment/version_0/' 
        if type_of_run == 'train':
            outfile.write(first_part + ' '.join(strings) + ' --save_prefix ' + save_prefix + ' &> ' + 'run_log_files/' + save_prefix + '_' + type_of_run +'.out\"\n')
        elif type_of_run == 'oversmooth_tune':
            checkpoint_file = get_last_epoch_checkpoint(experiment_folder)
            outfile.write(first_part + ' '.join(strings) + ' --save_prefix ' + save_prefix + ' --oversmooth_param 0.9 ' + ' --load_train --model_to_load ' + checkpoint_file  + ' &> ' + 'run_log_files/' + save_prefix + '_' + type_of_run +'.out\"\n')
        elif type_of_run == 'classify':
            checkpoint_file = get_last_epoch_checkpoint(experiment_folder)
            if args.no_len_penalty:
                outfile.write(first_part + checkpoint_file + ' ' + str(num_subsamples) + ' ' + str(classify_seq_set_len) + ' --len_penalty 0.0 ' + ' --save_prefix ' + save_prefix + ' &> ' + 'run_log_files/' + save_prefix + '_' + type_of_run +'.out\"\n')
            else:
                outfile.write(first_part + checkpoint_file + ' ' + str(num_subsamples) + ' ' + str(classify_seq_set_len) + ' ' + ' --save_prefix ' + save_prefix + ' &> ' + 'run_log_files/' + save_prefix + '_' + type_of_run +'.out\"\n')
        elif type_of_run == 'generate':
            checkpoint_file = get_last_epoch_checkpoint(experiment_folder)
            if args.no_len_penalty:
                outfile.write(first_part + checkpoint_file + ' --save_prefix ' + save_prefix + ' --len_penalty 0.0 --annot_file &> ' + 'run_log_files/' + save_prefix + '_' + type_of_run + '.out\"\n')
            else:
                outfile.write(first_part + checkpoint_file + ' --save_prefix ' + save_prefix + ' --annot_file &> ' + 'run_log_files/' + save_prefix + '_' + type_of_run + '.out\"\n')


