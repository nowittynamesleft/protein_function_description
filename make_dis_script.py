import sys
import numpy as np
from utils import get_last_epoch_checkpoint
import pickle


'''
Generates a set of scripts for first training the models with different hyperparameter settings,
then using the last checkpoints of those models to classify, and generate descriptions for a set of validation GO terms
'''

if __name__ == '__main__':
    type_of_run = sys.argv[1]
    prefix = sys.argv[2]
    hyperparam_list_fname = prefix + '_hparams.pckl'
    if type_of_run == 'train':
        num_sampled_hparam_sets = int(sys.argv[3])
    elif len(sys.argv) > 3:
        print('There should not be a number of sampled hparam sets for classify or generate settings. There should be a hyperparam list pickle that is loaded named: ' + hyperparam_list_fname)

    # Architecture search hyperparameter settings:
    '''
    hyperparam_dict = { 'num_encoder_layers': [1, 2, 3],
                        'num_decoder_layers': [1, 2, 3],
                        'dim_feedforward': [128, 256, 512],
                        'emb_size': [128, 256, 512],
                        'sigma': [0.25, 0.5],
                        'dropout': [0.0, 0.25, 0.5],
                        'seq_set_len': [16, 32]
                        }
    '''
    #full_data_architecture_search_num_encoder_layers-2_num_decoder_layers-2_dim_feedforward-512_emb_size-128_sigma-0.5_dropout-0.5_seq_set_len-32.out

    hyperparam_dict = { 'num_encoder_layers': [2],
                        'num_decoder_layers': [2],
                        'dim_feedforward': [512],
                        'emb_size': [128],
                        'sigma': [0.5],
                        'dropout': [0.5],
                        'seq_set_len': [32]
                        }
    if type_of_run == 'train':
        hyperparam_sets = []
        for i in range(num_sampled_hparam_sets):
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
    if type_of_run == 'train':
        first_part = 'sbatch -N 1 -p gpu --gres=gpu:1 --mem 150G --wrap \"python train_and_test_model.py uniprot_sprot_annot_no_go_cutoff_randomly_split_train_set.csv uniprot_sprot_annot_no_go_cutoff_randomly_split_validation_set.csv '
    elif type_of_run == 'classify':
        first_part = 'sbatch -N 1 -p gpu --gres=gpu:1 --mem 150G --wrap \"python simple_classify.py random_split_first_1000_val_set.csv '
    elif type_of_run == 'generate':
        first_part = 'sbatch -N 1 -p gpu --gres=gpu:1 --mem 150G --wrap \"python simple_generate.py random_split_first_1000_val_set.csv '
    for hyperparam_set in hyperparam_sets:
        strings = ['--' + key + ' ' + str(value) for (key, value) in hyperparam_set.items()]
        save_prefix = prefix + '_' + '_'.join([key + '-' + str(value) for (key, value) in hyperparam_set.items()])
        experiment_folder = 'lightning_logs/' + save_prefix + '_experiment/version_0/' 
        if type_of_run == 'train':
            outfile.write(first_part + ' '.join(strings) + ' --save_prefix ' + save_prefix + ' &> ' + save_prefix + '_' + type_of_run +'.out\"\n')
        elif type_of_run == 'classify':
            checkpoint_file = get_last_epoch_checkpoint(experiment_folder)
            outfile.write(first_part + checkpoint_file + ' ' + str(num_subsamples) + ' ' + str(classify_seq_set_len) + ' ' + ' --save_prefix ' + save_prefix + ' &> ' + save_prefix + '_' + type_of_run +'.out\"\n')
        elif type_of_run == 'generate':
            checkpoint_file = get_last_epoch_checkpoint(experiment_folder)
            outfile.write(first_part + checkpoint_file + ' --save_prefix ' + save_prefix + '--annot_file &> ' + save_prefix + '_' + type_of_run + '.out\"\n')


