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


CHARS = ['R', 'X', 'S', 'G', 'W', 'I', 'Q', 'A', 'T', 'V', 'K', 'Y', 'C', 'N', 'L', 'F', 'D', 'M', 'P', 'H', 'E', 'U', 'O', 'B', 'Z', '-']
CHAR2IND = {c: CHARS.index(c) for c in CHARS}


class SequenceDataset(Dataset):
    """
    Sequence  centric way of getting samples; so a batch size of 64 would
    select 64 sequences. Uses fastas
    """
    def __init__(self, fasta_fname):
        id2seq = load_fasta(fasta_fname)
        
        self.prot_list = sorted(list(id2seq.keys()))
        self.seqs = np.array([seq2AAinds(id2seq[prot]) for prot in self.prot_list], dtype=object)
        self.alphabet = CHARS

    def __getitem__(self, prot_ind):
        
        return ([self.seqs[prot_ind]],) # to work with seq_go_collate_pad function

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
    def __init__(self, go_file, obo_file, num_samples, vocab=None, include_go=True, save_prefix='no_prefix'):
        self.read_annot_info(go_file)
        self.go_annot_mat = create_annot_mat(self.all_prot_ids, self.go_terms, self.go2prot_ids)
        self.init_obo_info(obo_file)
        self.tokenize_descriptions(self.go_desc_strings, vocab, save_prefix)
        
        self.alphabet = CHARS
        self.num_samples = num_samples
        self.collate_fn = partial(seq_go_collate_pad, seq_set_size=self.num_samples)
        self.include_go = include_go
        self.sample = True


    def read_annot_info(self, go_file):
        annot_df = pd.read_csv(go_file, sep='\t')
        prot_seq_rows = annot_df.apply(lambda row: row['Prot-seqs'].split(','), axis=1)
        prot_seq_rows = [[seq2AAinds(prot) for prot in prot_seq_row] for prot_seq_row in prot_seq_rows]
        self.go2seqs = dict(zip(annot_df['GO-term'], prot_seq_rows))
        self.prot_lists = [prots.split(',') for prots in annot_df['Prot-names']]
        self.go2prot_ids = dict(zip(annot_df['GO-term'], self.prot_lists))
        self.all_prot_ids = sorted(list(set([prot for prot_list in self.prot_lists for prot in prot_list])))

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
        id_to_depth_mf = dict(nx.single_target_shortest_path_length(graph, 'GO:0003674'))
        id_to_depth_bp = dict(nx.single_target_shortest_path_length(graph, 'GO:0008150'))
        id_to_depth_cc = dict(nx.single_target_shortest_path_length(graph, 'GO:0005575'))
        self.depths_of_go_terms = {}
        for go_term in self.go_terms:
            if go_term in id_to_depth_mf:
                self.depths_of_go_terms[go_term] = id_to_depth_mf[go_term]
            elif go_term in id_to_depth_bp:
                self.depths_of_go_terms[go_term] = id_to_depth_bp[go_term]
            elif go_term in id_to_depth_cc:
                self.depths_of_go_terms[go_term] = id_to_depth_cc[go_term]
            else:
                print(go_term + ' NOT FOUND IN OBO')


    def __getitem__(self, go_term_index):
        annotated_seqs = self.get_annotated_seqs(go_term_index)[0]
        if self.num_samples == len(annotated_seqs):
            selected_inds = np.random.choice(np.arange(len(annotated_seqs)), size=self.num_samples, replace=False)
        else:
            selected_inds = np.random.choice(np.arange(len(annotated_seqs)), size=self.num_samples)
        if self.sample:
            selected_seqs = np.array(annotated_seqs)[selected_inds]
        else:
            selected_seqs = np.array(annotated_seqs)
        if self.include_go: 
            return (selected_seqs, self.go_token_ids[go_term_index])
        else:
            return (selected_seqs,)

    def set_sample_mode(self, sample_mode):
        self.sample = sample_mode

    def set_include_go_mode(self, go_mode):
        self.include_go = go_mode

    def set_num_samples(self, num_samples):
        self.num_samples = num_samples

    def get_annotated_seqs(self, go_term_index):
        return (self.go2seqs[self.go_terms[go_term_index]],)

    def get_padded_descs(self):
        return pad_GO(self.go_token_ids)

    def __len__(self):
        return len(self.go_terms)


def seq_go_collate_pad(batch, seq_set_size=None):
    """
    Pads matrices of variable length
    Takes a batch_size-length list of (seq_set_size object numpy arrays, GO_len) tuples and 
    turns it into (batch_size, seq_set_size, alphabet_size, batch_max_len) PyTorch tensors.
    Switches the alphabet size and length to interface with pytorch conv1d layer.
    
    Batch size X 2 (if GO included) list of tuples
    first index of tuple is sequence set of batch, so seq_set_size X 
    """
    if len(batch[0]) == 2: # check whether there is a GO description set attached
        GO_present = True
    else:
        GO_present = False
    lengths = []
    if GO_present:
        go_desc_lengths = []
        for i, (seq_set, go_term_desc) in enumerate(batch):
            go_desc_lengths.append(len(go_term_desc))
            lengths.append([])
            assert seq_set_size == len(seq_set)
            for j, seq in enumerate(seq_set):
                lengths[-1].append(len(seq))
        max_go_desc_length = max(go_desc_lengths)
    else:
        for i in range(len(batch)):
            seq_set = batch[i][0]
            assert seq_set_size == len(seq_set)
            lengths.append([])
            for j, seq in enumerate(seq_set):
                lengths[-1].append(len(seq))
     
    lengths = torch.tensor(lengths)
    max_len = torch.max(lengths)
    S_mask = torch.ones((len(batch), seq_set_size, max_len), dtype=bool)
    for i in range(len(batch)):
        for j in range(len(seq_set)):
            S_mask[i, j, :lengths[i][j]] = False


    S_padded = torch.zeros((len(batch), seq_set_size, max_len))

    # Sequence padding
    for seq_set_ind in range(len(batch)):
        curr_S_padded = S_padded[seq_set_ind]
        curr_seq_set_lengths = lengths[seq_set_ind]
        if GO_present:
            (seq_set, _) = batch[i]
        else:
            seq_set = batch[i][0]
        pad_seq_set(curr_S_padded, seq_set, curr_seq_set_lengths, max_len)
        
    # pad GO descriptions
    if GO_present:
        batch_go_descs = [torch.from_numpy(np.array(go_desc)) for (_, go_desc) in batch]
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
    
    return annot_mat

