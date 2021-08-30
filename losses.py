import torch
from torch import nn
import torch.nn.functional as F


class cluster_entropy(nn.Module):

    def __init__(self):
        super(cluster_entropy, self).__init__()

    def forward(self, H_i):
        avg_probs = torch.mean(H_i, dim=0) # average over samples in batch
        avg_probs = torch.clamp(avg_probs, min=1e-8) # so that probabilities are at the minimum 1e-8
        loss = torch.sum(avg_probs*torch.log(avg_probs))
        return loss


class modmax_loss(nn.Module):

    def __init__(self):
        super(semantic_clustering_loss, self).__init__()

    def forward(self, H_i, A_i):
        adj = A_i > 0
        #S_ij = torch.mm(H_i, H_i.t())
        k = torch.diag(torch.sum(adj, dim=0))
        B = adj - torch.mm(k, torch.transpose(k))/torch.sum(adj)
        epsilon = 1e-8
        class_mod_mat = torch.mm(torch.mm(torch.transpose(H_i), B), H_i)/torch.sum(adj)
        loss = torch.trace(class_mod_mat)

        return loss


