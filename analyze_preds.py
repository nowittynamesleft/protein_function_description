import numpy as np
import pickle
import sys
from data import SequenceGOCSVDataset


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
