import numpy as np
import pickle
import sys
from data import SequenceGOCSVDataset
import torch
from scipy.stats import pearsonr
from utils import annotation_correctness, specificity_preference, annotation_robustness


#TODO: need to have a function that takes in a model and zero-shot predictions 
# and returns the logits for the predicted GO terms.
'''
def get_word_logits(model, seq_sets, descriptions):
    for seq_set in seq_sets:
'''


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


if __name__ == '__main__':
    pred_dict = pickle.load(open(sys.argv[1],'rb'))
    x = SequenceGOCSVDataset(sys.argv[2], 'go.obo', 32)
    #compute_global_average_properties(pred_dict, x, k=10)
    prob_mat = pred_dict['all_term_preds']
    correct_go_inds = pred_dict['seq_set_go_term_inds']
    n = 4
    correctness = annotation_correctness(prob_mat, correct_go_inds)
    print('Annotation correctness: ' + str(correctness))
    robustness_score = annotation_robustness(prob_mat, 4, correct_go_inds)
    print('Annotation robustness (lower is better): ' + str(robustness_score))
    specificity_preference = specificity_preference(prob_mat, correct_go_inds, x.adj_mat)
    print('Specificity preference: ' + str(specificity_preference))
