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


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]


def micro_aupr(label, score):
    """Computing AUPR (micro-averaging)"""
    label = label.flatten()
    score = score.flatten()

    order = np.argsort(score)[::-1]
    label = label[order]

    P = np.count_nonzero(label)
    # N = len(label) - P

    TP = np.cumsum(label, dtype=float)
    PP = np.arange(1, len(label)+1, dtype=float)  # python

    x = np.divide(TP, P)  # recall
    y = np.divide(TP, PP)  # precision

    pr = np.trapz(y, x)

    return pr


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print('Creating directory ' + directory)
        os.makedirs(directory) 

def annotation_correctness(log_prob_mat, correct_go_inds):
    # correct_go_inds is a list of lists of go indices that are assigned to
    # a sequence set.
    correctness = 0
    num_seq_sets = log_prob_mat.shape[0]
    for seq_set_ind in range(num_seq_sets):
        row = log_prob_mat[seq_set_ind, :]
        correct_mask = torch.zeros_like(row, dtype=bool)
        correct_mask[correct_go_inds[seq_set_ind]] = True
        correct = row[correct_mask]
        incorrect = row[~correct_mask]
        row_correctness = 0
        for correct_score in correct:
            row_correctness += torch.sum(correct_score > incorrect)
        row_correctness = row_correctness/(len(correct)*len(incorrect))
        correctness += row_correctness
    correctness /= num_seq_sets

    return correctness


def specificity_preference(log_prob_mat, correct_go_inds, parent_graph):
    # need to calculate only over correct go inds...
    spec_preference = 0
    num_seq_sets = log_prob_mat.shape[0]
    num_seq_sets_with_parent_terms = 0
    for seq_set_ind in range(num_seq_sets):
        row = log_prob_mat[seq_set_ind, :]
        correct_mask = torch.zeros_like(row, dtype=bool)
        correct_mask[correct_go_inds[seq_set_ind]] = True
        correct = row[correct_mask]
        parents_correct = parent_graph[correct_mask, :][:, correct_mask]
        row_spec_preference = 0
        num_terms_with_parents = torch.sum(torch.any(parents_correct, dim=1))
        for ind, correct_score in enumerate(correct):
            parent_scores = correct[parents_correct[ind]] 
            if len(parent_scores) > 0:
                row_spec_preference += torch.sum(correct_score > parent_scores)/len(parent_scores)# if there are multiple direct parents, average them
        if num_terms_with_parents > 0:
            norm_row_spec_preference = row_spec_preference/num_terms_with_parents
            spec_preference += norm_row_spec_preference
            num_seq_sets_with_parent_terms += 1
    spec_preference /= num_seq_sets_with_parent_terms
    return spec_preference

'''
def annotation_robustness(log_prob_mat, n, correct_go_inds):
    # assumes log_prob_mat is n*number of GO terms X number of GO terms matrix,
    # where n is the number of subsamples per row
    # each n rows has the same GO terms assigned to them
    assert n > 1
    assert torch.equal(torch.tensor(correct_go_inds[0]), torch.tensor(correct_go_inds[n - 1]))
    score_difference = 0
    num_go_term_sets = int(log_prob_mat.shape[0]/n)
    for go_term_set_ind in range(num_go_term_sets):
        rows = log_prob_mat[go_term_set_ind*n:go_term_set_ind*n + n, :]
        correct_mask = torch.zeros_like(rows[0, :], dtype=bool)
        correct_mask[correct_go_inds[go_term_set_ind*n]] = True
        correct_scores = rows[:, correct_mask]
        # matrix of scores, columns should have values that are close
        total_difference = 0
        num_correct_terms = correct_scores.shape[1]
        assert n == correct_scores.shape[0]
        for column in range(correct_scores.shape[1]):
            curr_scores = correct_scores[:, column]
            for score_ind in range(n):
                score = curr_scores[score_ind]
                score_mask = torch.ones_like(curr_scores, dtype=bool)
                score_mask[score_ind] = False
                other_scores = curr_scores[score_mask]
                total_difference += torch.sum(torch.abs(score - other_scores)) # need to calculate the difference between current score and other scores
        curr_diff = total_difference/(num_correct_terms*(n*(n-1)))
        score_difference += curr_diff
    score_difference /= num_go_term_sets
    return score_difference
'''

def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp)
    ranks[tmp] = torch.arange(len(x))
    return ranks

def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    """Compute correlation between 2 1-D vectors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)
    
    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)


def annotation_robustness(log_prob_mat, n, correct_go_inds):
    # assumes log_prob_mat is n*number of GO terms X number of GO terms matrix,
    # where n is the number of subsamples per row
    # each n rows has the same GO terms assigned to them
    # pearson's rank correlation version
    assert n > 1
    assert torch.equal(torch.tensor(correct_go_inds[0]), torch.tensor(correct_go_inds[n - 1]))
    all_corr = 0
    num_go_term_sets = int(log_prob_mat.shape[0]/n)
    for go_term_set_ind in range(num_go_term_sets):
        rows = log_prob_mat[go_term_set_ind*n:go_term_set_ind*n + n, :]
        correct_mask = torch.zeros_like(rows[0, :], dtype=bool)
        correct_mask[correct_go_inds[go_term_set_ind*n]] = True
        correct_scores = rows[:, correct_mask]
        # matrix of scores, columns should have values that are close
        num_correct_terms = correct_scores.shape[1]
        if num_correct_terms > 1: # rank only makes sense when there is more than one score to compare
            assert n == correct_scores.shape[0]
            total_correlation = 0
            for score_vec_ind in range(correct_scores.shape[0]):
                score_vec = correct_scores[score_vec_ind, :]
                for score_vec_2_ind in range(correct_scores.shape[0]): 
                    score_vec_2 = correct_scores[score_vec_2_ind, :]
                    if score_vec_ind != score_vec_2_ind:
                        total_correlation += spearman_correlation(score_vec, score_vec_2)
            curr_avg_corr = total_correlation/(n*(n-1))
            all_corr += curr_avg_corr
    all_corr /= num_go_term_sets
    return all_corr


def test_metrics():
    ac_test_probs = torch.Tensor([[0.7, 0.3, 0.9, 0.4, 0.9], 
                                  [0.9, 0.9, 0.2, 0.9, 0.8], 
                                  [0.2, 0.9, 0.7, 0.2, 0.8], 
                                  [0.3, 0.9, 0.2, 0.8, 0.9],
                                  [0.3, 0.9, 0.9, 0.1, 0.9]])

    ac_correct_inds = torch.eye(5).bool()
    ac = annotation_correctness(ac_test_probs, ac_correct_inds)
    print('Annotation correctness test: ' + str(ac))
    assert ac == 0.5

    sp_test_probs = torch.Tensor([[0.2, 0.1, 0.1, 0.05, 0.05], 
                                  [0.1, 0.1, 0.1, 0.1, 0.1], 
                                  [0.1, 0.1, 0.15, 0.1, 0.1], 
                                  [0.1, 0.1, 0.1, 0.15, 0.1], 
                                  [0.1, 0.1, 0.1, 0.1, 0.1]])

    sp_correct_inds = torch.triu(torch.ones((5,5))).bool()
    sp_parent_graph = torch.diag(torch.tensor([1, 1, 1, 1]), 1).bool()
    sp = specificity_preference(sp_test_probs, sp_correct_inds, sp_parent_graph)
    print('Specificity preference test: ' + str(sp))
    assert sp == 0.5

    ar_test_probs = torch.Tensor([[0.2, 0.3, 0.5], 
                                  [0.1, 0.5, 0], 
                                  [0.1, 0.2, 0], 
                                  [0.5, 0.3, 0], 
                                  
                                  [0.4, 0.2, 0], 
                                  [0.9, 1, 0], 
                                  [0.3, 0.4, 0.3], 
                                  [0.8, 0.7, 0.9]])
    n = 4
    ar_correct_go_inds = torch.Tensor([[1, 1, 0], 
                                       [1, 1, 0], 
                                       [1, 1, 0], 
                                       [1, 1, 0], 
                                       [0, 1, 1], 
                                       [0, 1, 1], 
                                       [0, 1, 1], 
                                       [0, 1, 1]]).bool()

    ar = annotation_robustness(ar_test_probs, n, ar_correct_go_inds)
    print('Annotation robustness test: ' + str(ar))
    assert ar == 0.0


if __name__ == '__main__':
    test_metrics()
