import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import pytorch_lightning as pl
from fasta_loader import load_fasta, seq2onehot, seq2AAinds
import pickle
import numpy as np
from torchtext.data.utils import get_tokenizer
import itertools
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from functools import partial
import pandas as pd
import obonet
import networkx as nx
from scipy.special import comb
import tqdm


CHARS = ['R', 'X', 'S', 'G', 'W', 'I', 'Q', 'A', 'T', 'V', 'K', 'Y', 'C', 'N', 'L', 'F', 'D', 'M', 'P', 'H', 'E', 'U', 'O', 'B', 'Z', '-']
CHAR2IND = {c: CHARS.index(c) for c in CHARS}


class SequenceDataset(Dataset):
    """
    Sequence  centric way of getting samples; so a batch size of 64 would
    select 64 sequences. Uses fastas
    """
    def __init__(self, fasta_fname, num_samples=1):
        id2seq = load_fasta(fasta_fname)
        
        self.prot_list = sorted(list(id2seq.keys()))
        self.seqs = np.array([seq2AAinds(id2seq[prot]) for prot in self.prot_list], dtype=object)
        self.alphabet = CHARS
        self.num_samples = num_samples

    def __getitem__(self, prot_ind):
        if self.num_samples != 1:
            if self.num_samples == len(self.seqs):
                selected_inds = np.random.choice(np.arange(len(self.seqs)), size=self.num_samples, replace=False)
            else:
                selected_inds = np.random.choice(np.arange(len(self.seqs)), size=self.num_samples)
            selected_seqs = np.array(self.seqs)[selected_inds]
            selected_prot_ids = np.array(self.prot_list)[selected_inds]
            return ([selected_prot_ids, selected_seqs]) # to work with seq_go_collate_pad function
        else: 
            return ([self.prot_list[prot_ind], self.seqs[prot_ind]]) # to work with seq_go_collate_pad function

    def __len__(self):
        return len(self.seqs)


class SequenceGOCSVDataset(Dataset):
    """
    Sequence GO Dataset class with descriptions.
    GO term centric way of getting samples; so a batch size of 64 would
    select 64 GO terms and sample num_samples sequences for each GO term,
    returning the chosen sequences and descriptions for each GO term

    CSV files are organized as follows:
    GO-term   GO-name    GO-def Prot-names  Prot-seqs 

    for each GO_ID:
        split sequences by comma, these are the list for that GO_ID

    """
    def __init__(self, go_file, obo_file, num_samples, vocab=None, include_go=True, subset_inds=None, save_prefix='no_prefix'):
        self.go_file = go_file
        self.obo_file = obo_file
        self.vocab = vocab
        self.read_annot_info(go_file, subset_inds=subset_inds)
        self.go_annot_mat, self.prot_id2annot_ind = create_annot_mat(self.all_prot_ids, self.go_terms, self.go2prot_ids)
        self.init_obo_info(obo_file)
        self.tokenize_descriptions(self.go_desc_strings, vocab, save_prefix)
        
        self.alphabet = CHARS
        self.num_samples = num_samples
        self.collate_fn = partial(seq_go_collate_pad, seq_set_size=self.num_samples)
        self.include_go = include_go
        self.include_all_valid_terms = False
        self.sample = True

    def read_annot_info(self, go_file, subset_inds=None):
        annot_df = pd.read_csv(go_file, sep='\t')
        if subset_inds is not None:
            annot_df = annot_df.iloc[subset_inds]
        prot_seq_rows = annot_df.apply(lambda row: row['Prot-seqs'].split(','), axis=1)
        prot_seq_rows = [[seq2AAinds(prot) for prot in prot_seq_row] for prot_seq_row in prot_seq_rows]
        self.go2seqs = dict(zip(annot_df['GO-term'], prot_seq_rows))
        self.prot_lists = [prots.split(',') for prots in annot_df['Prot-names']]
        self.go2prot_ids = dict(zip(annot_df['GO-term'], self.prot_lists))
        self.id2seq = {}
        for i in range(len(prot_seq_rows)):
            id2seq = dict(zip(self.prot_lists[i], prot_seq_rows[i]))
            self.id2seq.update(id2seq)
        self.all_prot_ids = sorted(self.id2seq.keys())

        self.go_terms = np.array(annot_df['GO-term'])
        self.go_names = np.array(annot_df['GO-name'])
        self.go_desc_strings = np.array(annot_df['GO-def'])


    def tokenize_descriptions(self, desc_strings, vocab, save_prefix):
        tokenizer = get_tokenizer('basic_english') 
        tokenized = [tokenizer(desc) for desc in desc_strings]
        # get vocab size -- what if it's just character by character?
        if vocab is None:
            self.vocab = sorted(list(set(itertools.chain.from_iterable(tokenized))))
            self.vocab.insert(0, '<SOS>')
            self.vocab.append('<EOS>')
            pickle.dump(self.vocab, open(save_prefix + '_vocab.pckl', 'wb'))
        else:
            self.vocab = vocab
        for token_list in tokenized:
            token_list.insert(0, '<SOS>')
            token_list.append('<EOS>')
        word_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        token_ids = [[word_to_id[token] for token in tokens_doc if token in word_to_id] for tokens_doc in tokenized] # remove unknown tokens
        self.go_descriptions = tokenized
        self.go_token_ids = token_ids


    def init_obo_info(self, obo_file):
        graph = obonet.read_obo(obo_file)
        adj_dict = dict(graph.adjacency())
        self.adj_mat = torch.zeros((len(self.go_terms), len(self.go_terms)), dtype=bool)
        for i, key in enumerate(self.go_terms):
            parents = adj_dict[key].keys()
            parent_inds = []
            for parent in parents:
                if parent in self.go_terms:
                    parent_inds.append(np.where(self.go_terms == parent)[0])
            #self.adj_mat[i, torch.tensor(parent_inds, dtype=torch.long)] = True
            if len(parent_inds) > 0:
                self.adj_mat[i, np.concatenate(parent_inds)] = True
        id_to_depth_mf = dict(nx.single_target_shortest_path_length(graph, 'GO:0003674'))
        id_to_depth_bp = dict(nx.single_target_shortest_path_length(graph, 'GO:0008150'))
        id_to_depth_cc = dict(nx.single_target_shortest_path_length(graph, 'GO:0005575'))
        self.depths_of_go_terms = {}
        self.branch_of_go_terms = {}
        for go_term in self.go_terms:
            if go_term in id_to_depth_mf:
                self.depths_of_go_terms[go_term] = id_to_depth_mf[go_term]
                self.branch_of_go_terms[go_term] = '<Branch-MF>'
            elif go_term in id_to_depth_bp:
                self.depths_of_go_terms[go_term] = id_to_depth_bp[go_term]
                self.branch_of_go_terms[go_term] = '<Branch-BP>'
            elif go_term in id_to_depth_cc:
                self.depths_of_go_terms[go_term] = id_to_depth_cc[go_term]
                self.branch_of_go_terms[go_term] = '<Branch-CC>'
            else:
                print(go_term + ' NOT FOUND IN OBO')

    
    def get_all_valid_term_mask(self, prot_ids):
        annot_inds = np.array([self.prot_id2annot_ind[prot_id] for prot_id in prot_ids])
        valid_go_term_mask = np.all(self.go_annot_mat[annot_inds], axis=0)
        return valid_go_term_mask
    
    def get_union_of_terms(self, prot_ids):
        annot_inds = np.array([self.prot_id2annot_ind[prot_id] for prot_id in prot_ids])
        union_go_term_mask = np.any(self.go_annot_mat[annot_inds], axis=0)
        return union_go_term_mask

    def get_identically_annotated_subsamples(self, go_term_index, num_sets, verbose=1):
        try:
            assert num_sets > 1
        except AssertionError:
            if verbose > 0:
                print('Warning: num_sets to subsample must be greater than 1 for robustness calculation to make sense.')
        # get multiple subsets of proteins that have the same GO terms in common within each set of proteins, given a single go term that you want the sets to have
        annotated_seqs = self.get_annotated_seqs(go_term_index)[0]
        annotated_prot_ids = self.get_annotated_prot_ids(go_term_index)[0] # only consider proteins containing the go term specified
        prots_with_terms = []
        num_terms = self.go_annot_mat.shape[1]
        try:
            assert len(annotated_prot_ids) > self.num_samples
        except AssertionError:
            print("Term selected does not have enough proteins to subset " + str(num_sets) + " sets. Skipping.")
            raise
        counter = 0
        while comb(len(prots_with_terms), self.num_samples) < num_sets: # first choose a set such that there are enough proteins with the terms such that you can actually choose num_sets sets
            # get one protein set with chosen GO term
            selected_inds = np.random.choice(np.arange(len(annotated_seqs)), size=self.num_samples, replace=False)
            # get all of the set's common terms
            common_term_mask = self.get_all_valid_term_mask(np.array(annotated_prot_ids)[selected_inds])
            # get all prot ids with at least those terms; they could have others
            prot_inds_with_terms, prots_with_terms = self.get_prot_ids_containing_go_mask(self.go_annot_mat, common_term_mask) # if you can't find enough proteins for the term set, try again
            common_terms = self.go_terms[common_term_mask.nonzero()[0]]
            counter += 1
            if counter > 1000:
                import ipdb; ipdb.set_trace()
        #print('Number of tries to get an appropriate protein set to choose ' + str(num_sets) + ':' + str(counter))
        #print(common_terms)
        '''
        # once there are enough protein sets to possibly sample, check what other terms the proteins have
        all_term_mask = self.get_union_of_terms(prots_with_terms)
        other_term_mask = np.logical_xor(all_term_mask, common_term_mask) # remove the terms common to all prots in set
        considered_annots = self.go_annot_mat[prot_inds_with_terms, :]
        '''
        # now do sampling of sets
        prot_id_sets = []
        seq_sets = []
        rejected = 0
        total_tries = 0
        while len(prot_id_sets) < num_sets:
            curr_prot_ids = np.random.choice(prots_with_terms, size=self.num_samples, replace=False).tolist()
            curr_common_term_mask = self.get_all_valid_term_mask(curr_prot_ids)
            if np.all(curr_common_term_mask == common_term_mask):
                for s in prot_id_sets:
                    if s == curr_prot_ids: # if the current set is the same as a previous one, reject it
                        rejected += 1
                        pass
                prot_id_sets.append(curr_prot_ids)
                seq_sets.append([self.id2seq[prot_id] for prot_id in curr_prot_ids])
            else:
                rejected += 1
                #import ipdb; ipdb.set_trace()
            total_tries += 1
        '''
        for prot_set_ind in range(num_sets):
            if sum(other_term_mask) == 0: # proteins all have the same terms, just sample
                curr_prot_ids = np.random.choice(prots_with_terms, size=self.num_samples, replace=False).tolist()
            else:
                not_unsafe = np.any(~considered_annots[:, other_term_mask], axis=1) # prot inds that have at least one "other" term absent
                not_unsafe_prot_inds = prot_inds_with_terms[not_unsafe]
                try:
                    assert len(not_unsafe_prot_inds) > 0
                except AssertionError:
                    import ipdb; ipdb.set_trace()
                selected_ind = np.random.choice(not_unsafe_prot_inds, size=1)
                curr_prot_ids = np.array(self.all_prot_ids)[selected_ind].tolist()
                curr_other_common_functions = (self.go_annot_mat[selected_ind, :].flatten())*other_term_mask # we only care about terms we want to remove
                while np.sum(curr_other_common_functions) > 0:
                    try:
                        assert len(curr_other_common_functions) == len(other_term_mask)
                    except AssertionError:
                        import ipdb; ipdb.set_trace()
                    # consider only those proteins that do not have at least one of the other common terms
                    considered_prot_inds_to_keep = np.all(~considered_annots[:, curr_other_common_functions], axis=1).nonzero()[0]
                    assert len(considered_prot_inds_to_keep) > 0 # if this isn't true, then there can't be any sets that have the common go inds and nothing more
                    temp_considered_prot_inds = prot_inds_with_terms[considered_prot_inds_to_keep]
                    additional_selected_ind = np.random.choice(temp_considered_prot_inds, size=1)
                    curr_other_common_functions = (self.go_annot_mat[additional_selected_ind, :].flatten())*curr_other_common_functions # elementwise multiply with previous common function vector to remove now not-in-common functions
            #if len(curr_prot_ids) < self.num_samples:
                #curr_prot_ids.extend(np.random.choice(prots_with_terms, size=(len(curr_prot_ids) - self.num_samples), replace=False).tolist()) # doesn't guarantee unique proteins
            prot_id_sets.append(curr_prot_ids)
            seq_sets.append([self.id2seq[prot_id] for prot_id in curr_prot_ids])
        '''
        try:
            assert self.have_same_common_terms(prot_id_sets, common_term_mask)
        except AssertionError:
            print('Not all sets chosen have common terms chosen!')
            import ipdb; ipdb.set_trace()

        #print('Length of protein sets chosen:')
        #print([len(prot_set) for prot_set in prot_id_sets])
        #print('Rejected sampled sets: ' + str(rejected))
        #print('Total tries: ' + str(total_tries))
        #print('Rejection rate:')
        #print(rejected/total_tries)
        
        return (prot_id_sets, seq_sets, common_terms, rejected, total_tries)
        

    def have_same_common_terms(self, prot_sets, common_term_mask):
        for prots in prot_sets:
            if np.any(common_term_mask != self.get_all_valid_term_mask(prots)):
                return False
        return True 

    def __getitem__(self, go_term_index):
        annotated_seqs = self.get_annotated_seqs(go_term_index)[0]
        annotated_prot_ids = self.get_annotated_prot_ids(go_term_index)[0]
        if self.num_samples == len(annotated_seqs):
            selected_inds = np.random.choice(np.arange(len(annotated_seqs)), size=self.num_samples, replace=False)
        else:
            selected_inds = np.random.choice(np.arange(len(annotated_seqs)), size=self.num_samples)
        if self.sample:
            selected_seqs = np.array(annotated_seqs, dtype=object)[selected_inds]
            selected_prot_ids = np.array(annotated_prot_ids)[selected_inds]
        else:
            selected_seqs = np.array(annotated_seqs)
            selected_prot_ids = np.array(annotated_prot_ids)
        if self.include_go: 
            return (selected_prot_ids, selected_seqs, self.go_token_ids[go_term_index])
        else:
            return (selected_prot_ids, selected_seqs)

    def set_sample_mode(self, sample_mode):
        self.sample = sample_mode

    def set_include_go_mode(self, go_mode):
        self.include_go = go_mode

    def set_num_samples(self, num_samples):
        self.num_samples = num_samples

    def get_annotated_seqs(self, go_term_index):
        return (self.go2seqs[self.go_terms[go_term_index]],)

    def get_annotated_prot_ids(self, go_term_index):
        return (self.go2prot_ids[self.go_terms[go_term_index]],)

    def get_prot_ids_containing_go_mask(self, annot_mat, go_mask):
        # okay, so I want a set of proteins that have at least these GO terms, not only exactly these GO terms.
        # the things that need to be identical are the protein sets' common GO terms, not the protein's GO terms.
        # So the solution is to have the proteins
        prot_inds = (annot_mat >= go_mask).all(axis=1).nonzero()[0] # go_annot_mat must have at least all the terms the go_mask has
        prot_id_set = np.array(self.all_prot_ids)[prot_inds]
        
        return prot_inds, prot_id_set.tolist()

    def get_prot_ids_exact_go_mask(self, annot_mat, go_mask):
        prot_inds = (annot_mat == go_mask).all(axis=1).nonzero()[0] # go_annot_mat must have  all the terms the go_mask has
        prot_id_set = np.array(self.all_prot_ids)[prot_inds]
        
        return prot_inds, prot_id_set.tolist()

    def get_prot_ids_with_at_least_one_go_in_mask(self, annot_mat, go_mask):
        prot_inds = (annot_mat[go_mask]).any(axis=1).nonzero()[0] # go_annot_mat must have  at least one of the terms in go mask
        prot_id_set = np.array(self.all_prot_ids)[prot_inds]
        
        return prot_inds, prot_id_set.tolist()

    def get_padded_descs(self):
        return pad_GO(self.go_token_ids)

    def __len__(self):
        return len(self.go_terms)


def seq_go_collate_pad(batch, seq_set_size=None):
    """
    Pads matrices of variable length
    Takes a batch_size-length list of (seq_set_size string arrays (prot IDs), seq_set_size object numpy arrays, GO_len) tuples and 
    turns it into (batch_size, seq_set_size, alphabet_size, batch_max_len) PyTorch tensors.
    Switches the alphabet size and length to interface with pytorch conv1d layer.
    
    Batch size X 2 (if GO included) list of tuples
    first index of tuple is sequence set of batch, so seq_set_size X 
    """
    if len(batch[0]) > 2: # check whether there is a GO description set attached
        GO_present = True
    else:
        GO_present = False
    lengths = []
    if GO_present:
        go_desc_lengths = []
        for i, (prot_id_set, seq_set, go_term_desc) in enumerate(batch):
            go_desc_lengths.append(len(go_term_desc))
            lengths.append([])
            assert seq_set_size == len(seq_set)
            for j, seq in enumerate(seq_set):
                lengths[-1].append(len(seq))
        max_go_desc_length = max(go_desc_lengths)
    else:
        for i in range(len(batch)):
            prot_id_set = batch[i][0]
            seq_set = batch[i][1]
            assert seq_set_size == len(seq_set)
            lengths.append([])
            for j, seq in enumerate(seq_set):
                lengths[-1].append(len(seq))
     
    lengths = torch.tensor(lengths)
    max_len = torch.max(lengths)
    S_mask = torch.ones((len(batch), seq_set_size, max_len), dtype=bool)
    for seq_set_ind in range(len(batch)):
        for j in range(seq_set_size):
            S_mask[seq_set_ind, j, :lengths[seq_set_ind][j]] = False

    S_padded = torch.zeros((len(batch), seq_set_size, max_len))

    # Sequence padding
    for seq_set_ind in range(len(batch)):
        curr_S_padded = S_padded[seq_set_ind]
        curr_seq_set_lengths = lengths[seq_set_ind]
        if GO_present:
            (prot_id_set, seq_set, _) = batch[seq_set_ind]
        else:
            (prot_id_set, seq_set) = batch[seq_set_ind]
        pad_seq_set(curr_S_padded, seq_set, curr_seq_set_lengths, max_len)
        
    # pad GO descriptions
    if GO_present:
        batch_go_descs = [torch.from_numpy(np.array(go_desc)) for (_, _, go_desc) in batch]
        GO_padded, GO_mask = pad_GO(batch_go_descs)
        return S_padded, S_mask, GO_padded, GO_mask
    else:
        return S_padded, S_mask


def pad_seq_set(S_padded, seq_set, lengths, max_len):
    for j in range(len(seq_set)):
        seq = seq_set[j]
        if max_len >= lengths[j]:
            S_padded[j][:lengths[j]] = torch.from_numpy(seq.transpose())
        else:
            S_padded[j][:max_len] = torch.from_numpy(seq[:max_len, :].transpose())


def pad_GO(go_descs):
    # takes tokenized GO descriptions and pads them to max of the list of descriptions
    go_desc_lengths = [len(desc) for desc in go_descs]
    GO_padded = torch.zeros((len(go_descs), max(go_desc_lengths)), dtype=torch.long)
    GO_mask = torch.ones(len(go_descs), max(go_desc_lengths), dtype=bool)
    for i in range(len(go_descs)):
        curr_go_desc = go_descs[i]
        curr_go_desc_length = go_desc_lengths[i]
        GO_mask[i, :curr_go_desc_length] = False
        for j, word in enumerate(curr_go_desc):
            GO_padded[i, j] = word
    return GO_padded, GO_mask


def create_annot_mat(prot_ids, go_terms, go2prot_ids):
    annot_mat = np.zeros((len(prot_ids), len(go_terms)), dtype=bool)
    prot_id2annot_ind = dict(zip(prot_ids, range(len(prot_ids))))
    for i, go_term in enumerate(go_terms):
        annotated_prot_inds = []
        for prot_id in go2prot_ids[go_term]:
            annotated_prot_inds.append(prot_id2annot_ind[prot_id])
        annot_mat[:, i][annotated_prot_inds] = True
    
    return annot_mat, prot_id2annot_ind


def test_identical_subsampling(test_csv, num_sets, seq_set_len):
    dataset = SequenceGOCSVDataset(test_csv, 'go.obo', seq_set_len)
    print('loaded dataset')
    rejection_rate = 0
    num_excluded_terms = 0
    common_terms_lists = []
    for term in tqdm.tqdm(range(0, len(dataset.go_terms))):
        try:
            (prot_id_sets, seq_sets, common_terms, rejected, total_tries) = dataset.get_identically_annotated_subsamples(term, num_sets)
            rejection_rate += rejected/total_tries
            common_terms_lists.append(common_terms)
        except AssertionError:
            num_excluded_terms += 1
            pass
    rejection_rate /= (len(dataset.go_terms) - num_excluded_terms)
    print('Rejection rate (average): ' + str(rejection_rate))
    num_terms_list = [len(term_list) for term_list in common_terms_lists]
    num_multiterm_selections = 0
    for num_terms in num_terms_list:
        if num_terms > 1:
            num_multiterm_selections += 1
    avg_num_common_terms = sum(num_terms_list)/len(common_terms_lists)
    print('Average number of common terms: ' + str(avg_num_common_terms))
    print('Number of common terms greater than 1: ' + str(num_multiterm_selections))
    import ipdb; ipdb.set_trace()
    

if __name__ == '__main__':
    test_identical_subsampling('uniprot_sprot_training_val_split.csv', 4, 16)
