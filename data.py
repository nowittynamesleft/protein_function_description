import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import pytorch_lightning as pl
from fasta_loader import load_fasta, seq2onehot
import pickle
import numpy as np

CHARS = ['R', 'X', 'S', 'G', 'W', 'I', 'Q', 'A', 'T', 'V', 'K', 'Y', 'C', 'N', 'L', 'F', 'D', 'M', 'P', 'H', 'E', 'U', 'O', 'B', 'Z', '-']


class SequenceGODataset(Dataset):
    """
    Sequence GO Dataset class with descriptions.
    GO term centric way of getting samples; so a batch size of 64 would
    select 64 GO terms and sample num_samples sequences for each GO term,
    returning the chosen sequences and descriptions for each GO term
    """
    def __init__(self, fasta_fname, keyword_file, num_samples):
        id2seq = load_fasta(fasta_fname)
        go_dict = pickle.load(open(keyword_file, 'rb'))
        self.annot_mat = np.array(go_dict['annot_mat'])
        self.go_terms = np.array(go_dict['go_terms'])
        self.go_descriptions = np.array(go_dict['go_descriptions'])
        self.prot_list = self.go_dict['prot_ids'].tolist()
        self.seqs = np.array([seq2onehot(id2seq[prot]) for prot in self.prot_list])
        self.num_samples = num_samples

    def __getitem__(self, go_term_index):
        annotated_prot_inds = np.where(self.annot_mat[:, go_term_index])[0]
        selected_inds = np.random.choice(annotated_prot_inds, size=self.num_samples)
        
        return (self.seqs[selected_inds], self.go_descriptions[go_term_index])

    def __len__(self):
        return len(self.go_terms)


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


class SequenceSetGODataset(Dataset):
    def __init__(self, fasta_fname, go_term_file):
        id2seq = load_fasta(fasta_fname)
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
    Takes a batch_size-length list of (protein_length, alphabet_size) numpy arrays and turns it into (batch_size, alphabet_size, length) PyTorch tensors
    Switches the alphabet size and length to interface with pytorch conv1d layer
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
    Takes a batch_size-length list of (protein_length, alphabet_size) numpy arrays and turns it into (batch_size, alphabet_size, length) PyTorch tensors
    Switches the alphabet size and length to interface with pytorch conv1d layer
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

    seq_kw_dataloader = DataLoader(seq_key_dataset, batch_size=batch_size, shuffle=True, collate_fn=seq_kw_collate_pad)
    return seq_kw_dataloader, seq_dim, keyword_vocab_size

# TODO: make data loader for length-transform code
