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
    def __init__(self, go_file, obo_file, num_samples, vocab=None, include_go=True):
        #id2seq = load_fasta(fasta_fname)
        annot_df = pd.read_csv(go_file, sep='\t')
        prot_seq_rows = annot_df.apply(lambda row: row['Prot-seqs'].split(','), axis=1)
        prot_seq_rows = [[seq2AAinds(prot) for prot in prot_seq_row] for prot_seq_row in prot_seq_rows]
        self.go2seqs = dict(zip(annot_df['GO-term'], prot_seq_rows))

        self.go_terms = np.array(annot_df['GO-term'])
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


        self.go_names = np.array(annot_df['GO-name'])
        self.go_desc_strings = np.array(annot_df['GO-def'])
        self.include_go = include_go
        self.sample = True
        print('Num go terms')
        print(len(self.go_terms))
        tokenizer = get_tokenizer('basic_english') 
        tokenized = [tokenizer(desc) for desc in annot_df['GO-def']]
        # get vocab size -- what if it's just character by character?
        if vocab is None:
            self.vocab = sorted(list(set(itertools.chain.from_iterable(tokenized))))
            self.vocab.insert(0, '<SOS>')
            self.vocab.append('<EOS>')
        else:
            self.vocab = vocab
        for token_list in tokenized:
            token_list.insert(0, '<SOS>')
            token_list.append('<EOS>')
            #print(token_list)
        word_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        print('<SOS> and <EOS> token numbers:')
        print(word_to_id['<SOS>'])
        print(word_to_id['<EOS>'])
        token_ids = [[word_to_id[token] for token in tokens_doc if token in word_to_id] for tokens_doc in tokenized] # remove unknown tokens
        '''
        one_hot_docs = [np.zeros((len(doc), len(self.vocab))) for doc in token_ids]
        for i, doc in enumerate(token_ids):
            for token_id in enumerate(doc):
                one_hot_docs[i][token_id] = 1
        '''
        self.go_descriptions = tokenized
        self.go_token_ids = token_ids
        #self.device = device
        
        self.prot_lists = [prots.split(',') for prots in annot_df['Prot-names']]
        #self.seqs = np.array([seq2onehot(id2seq[prot]) for prot in self.prot_list], dtype=object)
        self.alphabet = CHARS
        self.num_samples = num_samples
        self.collate_fn = partial(seq_go_collate_pad, seq_set_size=self.num_samples)

    def __getitem__(self, go_term_index):
        #annotated_prot_inds = np.where(self.annot_mat[:, go_term_index])[0]
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
    # TODO: decide whether the data will be one hot already or sequences...
    # No, it should be indices and not one hot so that the embedding layer can handle it easier
    # get sequence lengths and pad them
    # lengths = torch.tensor([t[0].shape[0] for t in batch]).to(device)
    #import ipdb; ipdb.set_trace()
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
    #S_padded[:, :seq_set_size, :] = len(CHARS) # add "no residue" entries in one-hot matrix

    # pad
    for i in range(len(batch)):
        if GO_present:
            (seq_set, _) = batch[i]
        else:
            seq_set = batch[i][0]
        for j in range(len(seq_set)):
            seq = seq_set[j]
            if max_len >= lengths[i, j]:
                S_padded[i, j][:lengths[i, j]] = torch.from_numpy(seq.transpose())
            else:
                S_padded[i, j][:max_len] = torch.from_numpy(seq[:max_len, :].transpose())
        
    # handle GO descriptions. Pad max length of the GO description?
    if GO_present:
        GO_padded = torch.zeros((len(batch), max_go_desc_length), dtype=torch.long)
    #GO_padded[:, :] = len(vocab) # padding token is last

        batch_go_descs = [torch.from_numpy(np.array(go_desc)) for (_, go_desc) in batch]
        batch_go_desc_lengths = [len(desc) for desc in batch_go_descs]
        GO_mask = torch.ones(len(batch), max_go_desc_length, dtype=bool)
        for i in range(len(batch)):
            curr_go_desc = batch_go_descs[i]
            curr_go_desc_length = batch_go_desc_lengths[i]
            GO_mask[i, :curr_go_desc_length] = False
            for j, word in enumerate(curr_go_desc):
                GO_padded[i, j] = word

        #return S_padded.to(device), S_mask.to(device), GO_padded.to(device), GO_mask.to(device)
        return S_padded, S_mask, GO_padded, GO_mask
    else:
        return S_padded, S_mask


class SequenceKeywordDataset(Dataset):
    def __init__(self, fasta_fname, keyword_file):
        id2seq = load_fasta(fasta_fname)
        self.keyword_dict = pickle.load(open(keyword_file, 'rb'))
        self.keyword_df = self.keyword_dict['keyword_df']
        self.keywords = np.array(self.keyword_df['keyword_inds'])
        self.seqs = np.array([seq2onehot(id2seq[prot]) for prot in self.keyword_df['Entry']])
        self.all_keywords = self.keyword_dict['all_keywords']
        self.prot_list = self.keyword_df['Entry'].tolist()

    def __getitem__(self, index):
        return (self.seqs[index], self.keywords[index])

    def __len__(self):
        return len(self.prot_list)


def previous_seq_kw_collate_pad(batch, device=None, max_len=1000):
    """
    Pads matrices of variable length
    Takes a batch_size-length list of (protein_length, alphabet_size) 
    numpy arrays and turns it into (batch_size, alphabet_size, length) PyTorch tensors.
    Switches the alphabet size and length to interface with pytorch conv1d layer.
    """
    # get sequence lengths
    #lengths = torch.tensor([t[0].shape[0] for t in batch]).to(device)
    print(batch)
    (seq_batch, keywords_list_batch) = batch
    lengths = torch.tensor([t.shape[0] for t in seq_batch]).to(device)
    S_padded = torch.zeros((len(batch), len(CHARS), max_len)).to(device)
    S_padded[:, len(CHARS) - 1, :] = 1 # add "no residue" entries in one-hot matrix

    # pad
    for i in range(len(batch)):
        if max_len >= lengths[i]:
            S_padded[i][:, :lengths[i]] = torch.from_numpy(batch[i].transpose())
        else:
            S_padded[i][:, :max_len] = torch.from_numpy(batch[i][:max_len, :].transpose())

    batch_keywords = [torch.from_numpy(np.array(keyword_inds)).to(device) for keyword_inds in keywords_list_batch]
    return (S_padded, batch_keywords)


def seq_kw_collate_pad(batch, max_len=1000):
    """
    Pads matrices of variable length
    Takes a batch_size-length list of (protein_length, alphabet_size) numpy arrays and 
    turns it into (batch_size, alphabet_size, length) PyTorch tensors.
    Switches the alphabet size and length to interface with pytorch conv1d layer.
    """
    # get sequence lengths
    #lengths = torch.tensor([t[0].shape[0] for t in batch]).to(device)
    lengths = []
    for (seq, keywords_list) in batch:
        lengths.append(seq.shape[0])
    lengths = torch.tensor(lengths)
    S_padded = torch.zeros((len(batch), len(CHARS), max_len))
    S_padded[:, len(CHARS) - 1, :] = 1 # add "no residue" entries in one-hot matrix

    # pad
    for i in range(len(batch)):
        (seq, _) = batch[i]
        if max_len >= lengths[i]:
            S_padded[i][:, :lengths[i]] = torch.from_numpy(seq.transpose())
        else:
            S_padded[i][:, :max_len] = torch.from_numpy(seq[:max_len, :].transpose())

    batch_keywords = [torch.from_numpy(np.array(keyword_inds)) for (_, keyword_inds) in batch]
    return S_padded, batch_keywords


def get_data_loader(fasta_fname, keyword_file, batch_size, device=None):
    seq_key_dataset = SequenceKeywordDataset(fasta_fname, keyword_file)

    print('First sequence one-hot shape')
    print(seq_key_dataset.seqs[0].shape)
    seq_dim = seq_key_dataset.seqs[0].shape[1]
    print('First keyword indices')
    print(seq_key_dataset.keywords[0])
    keyword_vocab_size = len(seq_key_dataset.all_keywords)
    print(keyword_vocab_size)

    seq_kw_dataloader = DataLoader(seq_key_dataset, batch_size=batch_size, 
            shuffle=True, collate_fn=seq_kw_collate_pad)
    return seq_kw_dataloader, seq_dim, keyword_vocab_size

# TODO: make data loader for length-transform code
