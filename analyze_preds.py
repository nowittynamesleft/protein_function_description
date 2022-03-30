import numpy as np
import pickle
import sys
from data import SequenceGOCSVDataset
import torch
from scipy.stats import pearsonr
from utils import annotation_correctness, specificity_preference, annotation_robustness, micro_AUPR


#TODO: need to have a function that takes in a model and zero-shot predictions 
# and returns the logits for the predicted GO terms.


def compute_oversmoothing_logratio(logits, target, non_pad_mask, eos_idx, margin=1e-5):
    # from Kulikov et al. 2021
    full_lprobs = torch.log_softmax(logits, dim=-1)
    target_lprobs = torch.gather(full_lprobs, dim=-1, index=target.unsqueeze(-1))

    # reverse cumsum fast workaround, this makes approximation error for suffix_lprob[:,-1]
    # in other words, after this operation the smallest suffix of one token does not equal eaxctly to that
    # true eos_probability. So it is better to exlcude those positions from OSL since theoretically loss there is 0.
    target_lprobs_withoutpad = (target_lprobs * non_pad_mask).squeeze(-1)
    suffix_lprob = target_lprobs_withoutpad + torch.sum(target_lprobs_withoutpad, dim=-1, keepdims=True) - torch.cumsum(target_lprobs_withoutpad, dim=-1)
    
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

    return oversmoothing_loss, osr


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
    print('Annotation robustness (avg. spearman correlation between rankings. [-1, 1]): ' + str(robustness_score))
    return aupr, correctness, sp, robustness_score

def get_attributes(pred_fname, go_adj_mat):
    pred_dict = pickle.load(open(pred_fname, 'rb'))
    #compute_global_average_properties(pred_dict, x, k=10)
    prob_mat = pred_dict['all_term_preds']
    try:
        correct_go_mask = pred_dict['seq_set_go_term_mask']
    except KeyError:
        correct_go_mask = pred_dict['seq_set_go_term_inds']
    n = 4
    return attribute_calculation(prob_mat, correct_go_mask, n, go_adj_mat)

def pred_list_attributes(pred_fnames, go_adj_mat):
    auprs = []
    correctness_scores = []
    sps = []
    robustness_scores = []
    for fname in pred_fnames:
        aupr, correctness, sp, robustness_score = get_attributes(fname, go_adj_mat) 
        auprs.append(aupr)
        correctness_scores.append(correctness)
        sps.append(sp)
        robustness_scores.append(robustness_score)
    print(auprs)
    print(correctness_scores)
    print(sps)
    print(robustness_scores)

if __name__ == '__main__':
    pred_fnames = sys.argv[1:-1]
    dataset_fname = sys.argv[-1]
    dataset = SequenceGOCSVDataset(dataset_fname, 'go.obo', 32)
    go_adj_mat = dataset.adj_mat
    pred_list_attributes(pred_fnames, go_adj_mat)
