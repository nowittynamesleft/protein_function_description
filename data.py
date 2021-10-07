import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import pytorch_lightning as pl
from fasta_loader import load_fasta, seq2onehot
import pickle
import numpy as np
from torchtext.data.utils import get_tokenizer
import itertools
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from functools import partial


CHARS = ['R', 'X', 'S', 'G', 'W', 'I', 'Q', 'A', 'T', 'V', 'K', 'Y', 'C', 'N', 'L', 'F', 'D', 'M', 'P', 'H', 'E', 'U', 'O', 'B', 'Z', '-']
CHAR2IND = {c: CHARS.index(c) for c in CHARS}


class SequenceGODataset(Dataset):
    """
    Sequence GO Dataset class with descriptions.
    GO term centric way of getting samples; so a batch size of 64 would
    select 64 GO terms and sample num_samples sequences for each GO term,
    returning the chosen sequences and descriptions for each GO term
    """
    def __init__(self, fasta_fname, go_file, num_samples):
        id2seq = load_fasta(fasta_fname)
        go_dict = pickle.load(open(go_file, 'rb'))
        self.annot_mat = np.array(go_dict['annot'])
        self.go_terms = np.array(go_dict['go_terms'])
        tokenizer = get_tokenizer('basic_english') 
        tokenized = [tokenizer(desc) for desc in go_dict['descriptions']]
        # get vocab size -- what if it's just character by character?
        self.vocab = list(set(itertools.chain.from_iterable(tokenized)))
        word_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        token_ids = [[word_to_id[token] for token in tokens_doc] for tokens_doc in tokenized]
        print(token_ids)
        one_hot_docs = [np.zeros((len(doc), len(self.vocab))) for doc in token_ids]
        for i, doc in enumerate(token_ids):
            for token_id in enumerate(doc):
                one_hot_docs[i][token_id] = 1
        self.go_descriptions = one_hot_docs
        
        self.prot_list = go_dict['prot_ids']
        self.seqs = np.array([seq2onehot(id2seq[prot]) for prot in self.prot_list], dtype=object)
        self.num_samples = num_samples
        self.collate_fn = partial(seq_go_collate_pad, seq_set_size=self.num_samples, vocab_size=len(self.vocab))

    def __getitem__(self, go_term_index):
        annotated_prot_inds = np.where(self.annot_mat[:, go_term_index])[0]
        selected_inds = np.random.choice(annotated_prot_inds, size=self.num_samples)
        
        return (self.seqs[selected_inds], self.go_descriptions[go_term_index])

    def __len__(self):
        return len(self.go_terms)


def seq_go_collate_pad(batch, seq_set_size=None, vocab_size=None, device=None):
    """
    Pads matrices of variable length
    Takes a batch_size-length list of (seq_set_size, alphabet_size) object numpy arrays and 
    turns it into (batch_size, seq_set_size, alphabet_size, batch_max_len) PyTorch tensors.
    Switches the alphabet size and length to interface with pytorch conv1d layer.
    """
    # TODO: decide whether the data will be one hot already or sequences...
    # if it's one hot already, is that most efficient? probably. already is from the dataset class
    # get sequence lengths
    # lengths = torch.tensor([t[0].shape[0] for t in batch]).to(device)
    lengths = []
    go_desc_lengths = []
    for seq_set, go_term_desc in batch:
        go_desc_lengths.append(len(go_term_desc))
        lengths.append([])
        for seq in seq_set:
            lengths[-1].append(len(seq))
    lengths = torch.tensor(lengths)
    print('Hello world')
    print(lengths)
    max_len = torch.max(lengths)
    print(max_len)
    max_go_desc_length = max(go_desc_lengths)

    S_padded = torch.zeros((len(batch), seq_set_size, len(CHARS), max_len))
    S_padded[:, :seq_set_size, len(CHARS) - 1, :] = 1 # add "no residue" entries in one-hot matrix

    GO_padded = torch.zeros((len(batch), vocab_size, max_go_desc_length))
    GO_padded[:, vocab_size - 1, :] = 1 # add "no residue" entries in one-hot matrix
    # pad
    for i in range(len(batch)):
        (seq_set, _) = batch[i]
        for j in range(len(seq_set)):
            seq = seq_set[j]
            if max_len >= lengths[i, j]:
                S_padded[i, j][:, :lengths[i, j]] = torch.from_numpy(seq.transpose())
            else:
                S_padded[i, j][:, :max_len] = torch.from_numpy(seq[:max_len, :].transpose())
        

    # handle GO descriptions. Pad max length of the GO description?
    batch_go_descs = [torch.from_numpy(np.array(go_desc)) for (_, go_desc) in batch]
    return S_padded, batch_go_descs


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