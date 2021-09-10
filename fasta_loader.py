import torch
import numpy as np
from torch.utils import data
from Bio import SeqIO

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
#device = torch.device("cuda:0" if use_cuda else "cpu")

path = '/mnt/ceph/users/dberenberg/Data/cath/'
CHARS = ['R', 'X', 'S', 'G', 'W', 'I', 'Q', 'A', 'T', 'V', 'K', 'Y', 'C', 'N', 'L', 'F', 'D', 'M', 'P', 'H', 'E', 'U', 'O', 'B', 'Z', '-']


def load_fasta(filename):
    """ Loads fasta file and returns a dictionary of sequences """
    domain2seq = {}
    for entry in SeqIO.parse(open(filename, 'r'), 'fasta'):
        seq = str(entry.seq)
        entry = str(entry.id)
        #entry = entry.split('|')[-1]
        #entry = entry.split('/')[0]
        entry = entry.split('|')[1]
        domain2seq[entry] = seq
    return domain2seq

domain2seqres = load_fasta(path + 'materials/' + 'cath-dataset-nonredundant-S40.fa')


def load_domain_list(filename):
    """ Load list of CATH domain names """
    l = []
    fRead = open(filename, 'r')
    for line in fRead:
        l.append(line.strip())
    fRead.close()
    return l


def seq2onehot(seq, sub=None):
    """ Create 22-dim 1-hot embedding """
    vocab_size = len(CHARS)
    vocab_embed = dict(zip(CHARS, range(vocab_size)))

    # Convert vocab to one-hot
    if sub is None:
        extract = lambda x: vocab_embed[x] 
    else:
        extract = lambda x: vocab_embed.get(x, None) or vocab_embed[sub]

    vocab_one_hot = np.eye(vocab_size)
    embed_x = [extract(v) for v in seq]
    seqs_x = np.array([vocab_one_hot[j, :] for j in embed_x])

    return seqs_x


def onehot2seq(S):
    chars = np.asarray(CHARS)
    rind = np.argmax(np.exp(S[0]), 1)
    seq = "".join(list(chars[rind]))
    return seq


def collate_pad(batch, device=None, max_len=1000):
    """
    Pads matrices of variable length
    Takes a batch_size-length list of (protein_length, alphabet_size) numpy arrays and turns it into (batch_size, alphabet_size, length) PyTorch tensors
    Switches the alphabet size and length to interface with pytorch conv1d layer
    """
    # get sequence lengths
    #lengths = torch.tensor([t[0].shape[0] for t in batch]).to(device)
    lengths = torch.tensor([t.shape[0] for t in batch]).to(device)
    #max_len = max(lengths)
    S_padded = torch.zeros((len(batch), len(CHARS), max_len)).to(device)
    S_padded[:, len(CHARS) - 1, :] = 1 # add "no residue" entries in one-hot matrix

    # pad
    '''
    for i in range(len(batch)):
        S_padded[i][:lengths[i], :] = batch[i][1]
    '''
    for i in range(len(batch)):
        if max_len >= lengths[i]:
            S_padded[i][:, :lengths[i]] = torch.from_numpy(batch[i].transpose())
        else:
            S_padded[i][:, :max_len] = torch.from_numpy(batch[i][:max_len, :].transpose())

    return S_padded


class Dataset(data.Dataset):
    """ Characterizes a dataset for PyTorch """
    def __init__(self, domain_IDs):
        'Initialization'
        self.domain_IDs = domain_IDs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.domain_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.domain_IDs[index]

        # Load data
        S = torch.from_numpy(seq2onehot(domain2seqres[ID])).float()

        return S


if __name__ == "__main__":
    domains = load_domain_list(path + 'cath-dataset-nonredundant-S40.list')
    # Parameters
    params = {'batch_size': 64,
              'shuffle': True,
              'collate_fn': collate_padd}
    training_dataset = Dataset(domains[:1000])
    training_generator = data.DataLoader(training_dataset, **params)

    # for epoch in range(10):
    for batch in training_generator:
        print (torch.diagonal(batch[0][0]))
