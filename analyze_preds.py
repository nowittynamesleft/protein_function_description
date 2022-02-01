import numpy as np
import pickle
import sys
from data import SequenceGOCSVDataset


#TODO: need to have a function that takes in a model and zero-shot predictions 
# and returns the logits for the predicted GO terms.
def get_word_logits(model, seq_sets, descriptions):
    for seq_set in seq_sets:


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
    rankings = np.argsort(pred_dict['all_term_preds'], axis=-1)[::-1] # descending order
    predicted_terms = rankings[:, 0]
    avg_top_rankings = np.mean(rankings, axis=0)

    # average properties for terms in the dataset
    top_avg_terms = np.argsort(avg_top_rankings)

    # calculate average description length of terms
    go_descs = np.array(dataset.go_descriptions)
    predicted_tokenizations = go_descs[predicted_terms]
    predicted_descriptions = tokenizations_to_descriptions(predicted_tokenizations)

    top_avg_go_term_descs = go_descs[top_avg_terms]
    top_go_term_desc_lengths = [len(desc) for desc in top_avg_go_term_descs]
    print('Top '  + str(k) + ' lengths: ' +str(top_go_term_desc_lengths[:k]))

    # calculate average depth of terms
    top_avg_go_term_depths = np.array(dataset.depths_of_go_terms[dataset.go_terms[i]] for i in top_avg_terms)
    print('Top '  + str(k) + ' depths in GO tree: ' + str(top_go_term_desc_lengths[:k]))

    # calculate average ranking of words in the list of predictions
    # for a given word, what was the average rank of the description it was in?
    '''
    word_rank_sums = {}
    total_occurences = {}
    for rank, desc in enumerate(go_descs[top_avg_terms]):
        for token in desc:
            if token in word_rank_sums:
                word_rank_sums[token] += rank
                total_occurences[token] += 1
            else:
                word_rank_sums[token] = rank
                total_occurences[token] = 1
    top_word_list = sorted(word_rank_sums.keys(), key=lambda word: word_rank_sums[word]/total_occurences[word])
    top_word_rankings = [word_rank_sums[word]/total_occurences[word] for word in top_word_list]
    '''

    import ipdb; ipdb.set_trace()
    occurences_in_top_k = {}
    for i in range(0, k):
        desc = go_descs[top_avg_terms[i]]
        for token in desc:
            if token in occurences_in_top_k:
                occurences_in_top_k[token] += 1
            else:
                occurences_in_top_k[token] = 1
    most_common_words_in_top_k_go_terms = sorted(occurences_in_top_k.keys(), key=lambda word: occurences_in_top_k[word]) 
    print('Done')


     

if __name__ == '__main__':
    pred_dict = pickle.load(open(sys.argv[1],'rb'))
    x = SequenceGOCSVDataset(sys.argv[2], 'go.obo', 32)
    compute_global_average_properties(pred_dict, x, k=10)