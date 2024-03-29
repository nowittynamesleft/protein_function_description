import numpy as np
import pickle
import sys
from data import SequenceGOCSVDataset
import torch
from scipy.stats import pearsonr
from utils import annotation_correctness, specificity_preference, annotation_robustness, micro_AUPR, get_pairwise_rank_correlations
import argparse


def arguments():
    args = argparse.ArgumentParser()
    args.add_argument('annot_seq_file', type=str, help='Annotation dataset corresponding to prediction files')
    args.add_argument('--pred_files', type=str, nargs='+')
    args.add_argument('--no_input_files', type=str, nargs='+')
    args.add_argument('--num_subsamples', type=int, default=1)
    args = args.parse_args()
    print(args)
    return args


def compute_oversmoothing_logratio(logits, target, non_pad_mask, eos_idx, margin=1e-5):
    # from Kulikov et al. 2021
    full_lprobs = torch.log_softmax(logits, dim=-1)
    target_lprobs = torch.gather(full_lprobs, dim=-1, index=target.unsqueeze(-1))

    # reverse cumsum fast workaround, this makes approximation error for suffix_lprob[:,-1]
    # in other words, after this operation the smallest suffix of one token does not equal eaxctly to that
    # true eos_probability. So it is better to exclude those positions from OSL since theoretically loss there is 0.
    target_lprobs_withoutpad = (target_lprobs * non_pad_mask).squeeze(-1)
    suffix_lprob = target_lprobs_withoutpad + torch.sum(target_lprobs_withoutpad, dim=-1, keepdims=True) - torch.cumsum(target_lprobs_withoutpad, dim=-1)
    
    #import ipdb; ipdb.set_trace()
    eos_lprobs = full_lprobs[:,:,eos_idx] * non_pad_mask.squeeze(-1)

    oversmoothing_loss = torch.maximum(eos_lprobs - suffix_lprob + margin, torch.zeros_like(suffix_lprob))
    oversmoothing_loss = (oversmoothing_loss.sum(dim=1) / non_pad_mask.squeeze(dim=-1).sum(dim=1)).mean()

    # computing the oversmoothing rate here for free
    with torch.no_grad():
        oversmoothed = eos_lprobs > suffix_lprob
        oversmoothed = oversmoothed * non_pad_mask.squeeze(-1)  # exclude pad cases from oversmoothing rate
        oversmoothed = oversmoothed * (target != eos_idx).float() # exclude t=true_eos from oversmoothing counts

        num_osr_per_seq = non_pad_mask.squeeze(-1).sum(-1) - 1  # exclude the <eos> from each seq count
        osr = oversmoothed.sum(-1) / num_osr_per_seq # compute oversmoothing per sequence
        avg_osr = osr.mean()

    return oversmoothing_loss, avg_osr


def tokenization_to_description(tokenized_desc): # assume @@ is BPE separator
    return [''.join([token[:-2] if token[-2:] == '@@' else token + ' ' for token in tokenized_desc])]

def tokenizations_to_descriptions(tokenized_descs): # assume @@ is BPE separator
    return [tokenization_to_description(desc) for desc in tokenized_descs]

def compute_global_average_properties(pred_dict, dataset, k=10):
    # get rankings for each seq set and find average rankings of each term for whole dataset
    rankings = torch.sort(pred_dict['all_term_preds'], dim=1)[1] + 1
    recip_rankings = 1/rankings
    relevant_recip_ranks = recip_rankings[torch.arange(0, rankings.shape[0]), pred_dict['seq_set_go_term_inds']]
    mrr = relevant_recip_ranks.mean()

    # calculate average description length of terms
    go_descs = np.array(dataset.go_descriptions)

    lengths = [len(go_descs[ind]) for ind in pred_dict['seq_set_go_term_inds']]
    depths = [dataset.depths_of_go_terms[dataset.go_terms[i]] for i in pred_dict['seq_set_go_term_inds']]

    corr_length = pearsonr(relevant_recip_ranks, lengths)
    corr_depth = pearsonr(relevant_recip_ranks, depths)
    print('Correlation between relevant reciprocal rank and length of description')
    print(corr_length)
    print('Correlation between relevant reciprocal rank and depth of term')
    print(corr_depth)

def attribute_calculation(prob_mat, correct_go_mask, n, go_adj_mat):
    aupr = micro_AUPR(np.stack(correct_go_mask), prob_mat.numpy())
    print('AUPR: ' + str(aupr))
    correctness = annotation_correctness(prob_mat, correct_go_mask)
    print('Annotation correctness: ' + str(correctness))
    sp = specificity_preference(prob_mat, correct_go_mask, go_adj_mat)
    print('Specificity preference: ' + str(sp))
    robustness_score = annotation_robustness(prob_mat, n, correct_go_mask)
    print('Annotation robustness (avg. spearman correlation between rankings. [-1, 1]): ' + str(robustness_score) + '\n')
    
    return aupr, correctness, sp, robustness_score

def get_attributes(pred_fname, go_adj_mat, n, go_term_mask=None, no_input_file=None, freq_adj_param=1.0, annot_sums=None):
    pred_dict = pickle.load(open(pred_fname, 'rb'))
    if no_input_file is not None:
        score_dict = pickle.load(open(no_input_file, 'rb'))
        desc_background_log_probs = score_dict['desc_scores'].cpu()
        desc_background_log_probs = desc_background_log_probs/torch.tensor([len(desc) for desc in score_dict['go_descriptions']])
    else:
        desc_background_log_probs = torch.zeros((len(pred_dict['go_descriptions']),))
    try:
        correct_go_mask = np.stack(pred_dict['seq_set_go_term_mask'])
    except KeyError:
        correct_go_inds = pred_dict['seq_set_go_term_inds']
        correct_go_mask = np.zeros_like(prob_mat)
        for correct_go_ind_row in correct_go_inds:
            correct_go_mask[correct_go_ind_row, :] = True

    if go_term_mask is not None:
        correct_go_mask = correct_go_mask[:, go_term_mask]
        prob_mat = pred_dict['all_term_preds'][:, go_term_mask]
        go_adj_mat = go_adj_mat[go_term_mask, :][:, go_term_mask]
        # need to remove zero rows
        nonzero_rows = correct_go_mask.sum(axis=1).nonzero()[0]
        print("Number of nonzero_rows " + str(len(nonzero_rows)))
        print("Number of GO terms considered " + str(sum(go_term_mask)))
        correct_go_mask = correct_go_mask[nonzero_rows, :]
        prob_mat = prob_mat[nonzero_rows, :]
    else:
        print('No filtered subset of GO terms')
        prob_mat = pred_dict['all_term_preds']
    #avg_corr = get_pairwise_rank_correlations(prob_mat)
    #print('Average rank correlation: ' + str(avg_corr))
    #aupr, correctness, sp, robustness_score = attribute_calculation(prob_mat, correct_go_mask, n, go_adj_mat)

    if annot_sums is not None:
        print('Naive method based on just the frequency of this dataset:')
        #return attribute_calculation(torch.tensor(correct_go_mask.sum(axis=0)).repeat(correct_go_mask.shape[0], 1), correct_go_mask, n, go_adj_mat)
        return attribute_calculation(torch.tensor(annot_sums).repeat(correct_go_mask.shape[0], 1), correct_go_mask, n, go_adj_mat)

    if freq_adj_param != 0:
        print('Adjusting to get p(x|y) scores with adjustment parameter ' + str(freq_adj_param) + ', returning those as actual performances')
    else:
        print('No frequency adjustment')

    probabilities = np.exp(prob_mat.numpy())
    avg_probs = probabilities.sum(axis=0)/probabilities.shape[0]
    log_avg_probs = np.log(avg_probs)
    new_prob_mat = prob_mat - freq_adj_param*log_avg_probs.reshape(1, -1)
    avg_corr = get_pairwise_rank_correlations(new_prob_mat)

    print('Average rank correlation of prob mat with subtracted log p(y): ' + str(avg_corr))
    aupr, correctness, sp, robustness_score = attribute_calculation(new_prob_mat, correct_go_mask, n, go_adj_mat)

    if no_input_file is not None:
        no_input_adjusted_prob_mat = prob_mat - desc_background_log_probs.reshape(1, -1)
        avg_corr = get_pairwise_rank_correlations(no_input_adjusted_prob_mat)

        print('Average rank correlation of prob mat with subtracted log probs given with p(y|empty string): ' + str(avg_corr))
        attribute_calculation(no_input_adjusted_prob_mat, correct_go_mask, n, go_adj_mat)
    print('\n')
    return aupr, correctness, sp, robustness_score

def pred_list_attributes(pred_fnames, go_adj_mat, go_term_mask=None, num_subsamples=4, no_input_files=None, annot_sums=None):
    auprs = []
    correctness_scores = []
    sps = []
    robustness_scores = []
    import ipdb; ipdb.set_trace()
    for i in range(len(pred_fnames)):
        fname = pred_fnames[i] 
        if no_input_files is not None:
            no_input_file = no_input_files[i]
        else:
            no_input_file = None
        print(fname)
        #freq_adj_params = np.arange(0.8, 2, step=0.1)
        print('naive perfs')
        aupr, correctness, sp, robustness_score = get_attributes(fname, go_adj_mat, num_subsamples, go_term_mask=go_term_mask, no_input_file=no_input_file, freq_adj_param=0.0, annot_sums=annot_sums)
        print('perfs iterating through different frequency adjustment params')
        #for freq_adj_param in freq_adj_params:
        #    aupr, correctness, sp, robustness_score = get_attributes(fname, go_adj_mat, num_subsamples, go_term_mask=go_term_mask, no_input_file=no_input_file, freq_adj_param=freq_adj_param)
        aupr, correctness, sp, robustness_score = get_attributes(fname, go_adj_mat, num_subsamples, go_term_mask=go_term_mask, no_input_file=no_input_file, freq_adj_param=1.0)
        auprs.append(aupr)
        correctness_scores.append(correctness)
        sps.append(sp)
        robustness_scores.append(robustness_score)
    print(auprs)
    print(correctness_scores)
    print(sps)
    print(robustness_scores)

if __name__ == '__main__':
    args = arguments()
    dataset_fname = args.annot_seq_file
    pred_fnames = args.pred_files
    no_input_fnames = args.no_input_files
    num_subsamples = args.num_subsamples

    dataset = SequenceGOCSVDataset(dataset_fname, 'go.obo', 32)
    go_adj_mat = dataset.adj_mat
    annot_sums = dataset.go_annot_mat.sum(axis=0)
    print('Most annotated go term has ' + str(max(annot_sums)))
    print('Min annotated go term has ' + str(min(annot_sums)))
    min_examples = 0
    #min_examples = 32
    #max_examples = 1280
    print('Cutoff chosen: ' + str(min_examples))
    if min_examples > 0:
        considered_terms_mask = (annot_sums > min_examples) & (annot_sums < max_examples)
        print('Min annotated go term considered for metrics has ' + str(min(annot_sums[considered_terms_mask])))
        print('Description associated with least annotated term that is considered: ')
        print(np.array(dataset.go_descriptions)[considered_terms_mask][np.argmin(annot_sums[considered_terms_mask])])
        print('Description associated with most annotated term that is considered: ')
        print(np.array(dataset.go_descriptions)[considered_terms_mask][np.argmax(annot_sums[considered_terms_mask])])
        pred_list_attributes(pred_fnames, go_adj_mat, go_term_mask=considered_terms_mask, num_subsamples=num_subsamples, no_input_files=no_input_fnames)
    else:
        pred_list_attributes(pred_fnames, go_adj_mat, num_subsamples=num_subsamples, no_input_files=no_input_fnames, annot_sums=annot_sums)
