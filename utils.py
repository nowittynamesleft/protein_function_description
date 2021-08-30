import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pickle
from fasta_loader import load_fasta, seq2onehot


def count_clusters(softmax_outputs_list):
    total_clusters = set()
    for org_preds in softmax_outputs_list:
        curr_org_clusters = set(np.argmax(org_preds, axis=1))
        num_filled_clusters = len(curr_org_clusters)
        print(num_filled_clusters)
        total_clusters = total_clusters.union(curr_org_clusters)
    num_clusters = len(total_clusters)
    return num_clusters


def masked_loss(out, label, mask):

    loss = F.cross_entropy(out, label, reduction='none')
    mask = mask.float()
    mask = mask / mask.mean()
    loss *= mask
    loss = loss.mean()
    return loss


def masked_acc(out, label, mask):
    # [node, f]
    pred = out.argmax(dim=1)
    correct = torch.eq(pred, label).float()
    mask = mask.float()
    mask = mask / mask.mean()
    correct *= mask
    acc = correct.mean()
    return acc


def row_wise_normalize(mat):
    # row-wise normalization of nxn matrix
    n1 = mat.shape[0]
    with np.errstate(divide='ignore'):
        row_sums_inv = 1.0/mat.sum(axis=1)
    row_sums_inv[np.isposinf(row_sums_inv)] = 0

    row_sums_inv = np.asarray(row_sums_inv).reshape(-1)
    row_sums_inv = sparse.spdiags(row_sums_inv, 0, n1, n1)
    norm_mat = row_sums_inv.dot(mat)

    return norm_mat


def convert_to_tensor(*args, device=None):
    if len(args) > 1:
        return [torch.from_numpy(arg).to(device).float() for arg in args]
    else:
        return torch.from_numpy(args[0]).to(device).float()


def compute_nmi(softmax_scores, y):
    preds = np.argmax(softmax_scores, axis=1)
    return nmi(y, preds)


def get_common_indices(annot_prots, string_prots):
    common_prots = list(set(string_prots).intersection(annot_prots))
    print ("### Number of prots in intersection:", len(common_prots))
    annot_idx = [annot_prots.index(prot) for prot in common_prots] # annot_idx is the array of indices in the annotation protein list of each protein common to both annotation and string protein lists
    string_idx = [string_prots.index(prot) for prot in common_prots] # same thing for string protein list

    return annot_idx, string_idx


def get_top_k_element_list(sim_mat, k):
    top_k = np.argpartition(sim_mat, sim_mat.shape[1] - k, axis=1)[:, -k:]
    return top_k


def get_individual_keyword_embeds(model, vocab_size, device):
    total_keyword_embeds = []
    for i in range(0, vocab_size):
        ind = torch.tensor(i, dtype=torch.long).unsqueeze(0).to(device)
        keyword_embed = model.keyword_embed([ind])
        keyword_embeds = nn.functional.normalize(keyword_embed)
        total_keyword_embeds.append(keyword_embeds)
    total_keyword_embeds = torch.cat(total_keyword_embeds, dim=0).detach().cpu().numpy() 
    return total_keyword_embeds


def sparse_dropout(x, rate, noise_shape):
    """

    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).byte()
    i = x._indices() # [2, 49216]
    v = x._values() # [49216]

    # [2, 4926] => [49216, 2] => [remained node, 2] => [2, remained node]
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

    out = out * (1./ (1-rate))

    return out


def dot(x, y, sparse=False):
    if sparse:
        res = torch.sparse.mm(x, y)
    else:
        res = torch.mm(x, y)

    return res

