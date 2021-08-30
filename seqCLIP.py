import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import networkx as nx
import numpy as np
import pickle
import pandas as pd
from scipy import sparse
from sklearn.datasets import make_classification
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.model_selection import train_test_split
from functools import partial
from sklearn.cluster import KMeans
from utils import count_clusters, row_wise_normalize, convert_to_tensor, compute_nmi, get_common_indices, get_top_k_element_list, get_individual_keyword_embeds
from fasta_loader import load_fasta, seq2onehot
#from compare_clusters_with_go_sim import align_preds_with_annots, cluster_func_pred, make_cluster_lists_from_cluster_preds
#from compare_with_brenda_df import compute_brenda_nmi
from torch.utils.tensorboard import SummaryWriter
from fasta_loader import collate_padd
from tqdm import tqdm
import argparse
from models import ProtEmbedding, KeywordEmbedding, seqCLIP
from losses import cluster_entropy, modmax_loss

RANDOM_STATE = 973


def make_multi_graph_dataset(n_graphs, n_samples, n_graph_feats, n_in_feats, n_classes, prop_known_sims=1.0, device=None):
    # making many graphs out of groups of samples
    X, y = make_classification(n_samples=n_samples*n_graphs, n_features=n_graph_feats + n_in_feats, n_informative=n_graph_feats + n_in_feats, n_redundant=0, n_repeated=0, n_classes=n_classes, random_state=RANDOM_STATE)
    onehot = OneHotEncoder(sparse=False)
    onehot.fit(y.reshape(-1,1))

    graph_feats = X[:, :n_graph_feats]
    feats = X[:, n_graph_feats:]
    list_of_sample_feats = np.split(feats, n_graphs)
    list_of_sample_graph_feats = np.split(graph_feats, n_graphs)
    list_of_sample_classes = np.split(y, n_graphs)
    intergraph_conns = []
    graphs = []
    for i in range(0, len(list_of_sample_feats)):
        graph = cosine_similarity(list_of_sample_graph_feats[i])
        graphs.append(graph)
        class_assignments = list_of_sample_classes[i]
        oh_class_assignments = onehot.transform(class_assignments.reshape(-1,1))
        curr_intergraph_block_row = []
        for other_class_assn in list_of_sample_classes:
            oh_other_class_assn = onehot.transform(other_class_assn.reshape(-1,1))
            intergraph = np.matmul(oh_class_assignments, oh_other_class_assn.transpose())
            subsample_mask = np.random.rand(intergraph.shape[0], intergraph.shape[1]) > prop_known_sims
            intergraph[subsample_mask] = 1.0/n_classes
            curr_intergraph_block_row.append(intergraph)
        intergraph_conns.append(curr_intergraph_block_row)
            

    train_inds, test_inds = train_test_split(np.arange(0, len(graphs)), random_state=RANDOM_STATE)
    train_graphs = np.array(graphs)[train_inds]
    test_graphs =  np.array(graphs)[test_inds]
    train_feats = np.array(list_of_sample_feats)[train_inds]
    test_feats = np.array(list_of_sample_feats)[test_inds]
    train_intergraph_conns = np.array(intergraph_conns)[train_inds, :, :][:, train_inds, :]
    train_test_intergraph_conns = np.array(intergraph_conns)[train_inds, :, :][:, test_inds, :]
    test_intergraph_conns = np.array(intergraph_conns)[test_inds, :, :][:, test_inds, :]
    train_y_list = np.array(list_of_sample_classes)[train_inds]
    test_y_list = np.array(list_of_sample_classes)[test_inds]

    (train_feats, train_graphs, test_feats, test_graphs, train_intergraph_conns, train_test_intergraph_conns, 
            test_intergraph_conns, train_y_list, test_y_list) = convert_to_tensor(train_feats, train_graphs, 
                    test_feats, test_graphs, train_intergraph_conns, train_test_intergraph_conns, 
                    test_intergraph_conns, train_y_list, test_y_list, device=device)
    return train_feats, train_graphs, test_feats, test_graphs, train_intergraph_conns, train_test_intergraph_conns, test_intergraph_conns, train_y_list, test_y_list


def compute_loss_all_pairs(net, feats_list_1, feats_list_2, graphs_1, graphs_2, intergraphs, loss_fn, train=False):
    # computes loss between all pairs of graphs 1 and 2
    # should there be batching for the graphs compared to graph 1?
    # how about leaving batches as just single pairs for now
    loss = 0
    for first_graph_ind in range(0, len(graphs_1)):
        features_1 = feats_list_1[first_graph_ind]
        graph_1 = graphs_1[first_graph_ind]
        for second_graph_ind in range(0, len(graphs_2)):
            features_2 = feats_list_2[second_graph_ind]
            graph_2 = graphs_2[second_graph_ind]
            R_12 = intergraphs[first_graph_ind, second_graph_ind, :, :]

            H_i = net(features_1)
            H_j = net(features_2)
            loss += loss_fn(H_i, H_j, graph_1, graph_2, R_12)
    return loss


def train_epoch_clip(net, seqs_list, keywords_list, vocab_size, optimizer, batch_size=64, device=None):
    # computes loss between all pairs of graphs 1 and 2
    # should there be batching for the graphs compared to graph 1?
    # how about leaving batches as just single pairs for now
    total_objective = 0
    loss_fn = nn.CrossEntropyLoss() # using it to compare batch seqs and their neighbors' similarities with ones matrix
    permutation = torch.randperm(len(seqs_list))
    all_seq_embeds = []
    all_keyword_embeds = []
    for i in tqdm(range(0, len(seqs_list), batch_size)):
        optimizer.zero_grad()
        indices = permutation[i:i+batch_size]
        batch_seqs = seqs_list[indices]

        batch_seqs = collate_padd(batch_seqs, device=device)
        batch_keywords = [torch.from_numpy(np.array(keyword_inds)).to(device) for keyword_inds in keywords_list[indices]]

        seq_embeds, keyword_embeds = net(batch_seqs, batch_keywords) # keyword_embeds (Nxk) are averaged over all assigned keywords for a protein
        seq_embeds = nn.functional.normalize(seq_embeds)
        keyword_embeds = nn.functional.normalize(keyword_embeds)
        all_seq_embeds.append(seq_embeds)
        all_keyword_embeds.append(keyword_embeds)

        curr_batch_size, embed_dim = seq_embeds.size()
        #similarity = torch.bmm(seq_embeds.view(curr_batch_size, 1, embed_dim), keyword_embeds.view(curr_batch_size, embed_dim, 1)).squeeze()
        similarity = torch.mm(seq_embeds, keyword_embeds.transpose(0,1)).squeeze()
        similarity *= torch.exp(net.temperature)
        '''
        numpy_sim = similarity.detach().cpu().numpy()
        try:
            assert (numpy_sim >= 0.).all() and (numpy_sim <= 1.).all()
        except AssertionError:
            print(numpy_sim[numpy_sim > 1.])
            print(numpy_sim[numpy_sim < 0.])
            exit()
        '''
        labels = torch.arange(start=0, end=similarity.shape[0], dtype=torch.long).to(device)
        loss = (loss_fn(similarity, labels) + loss_fn(similarity.transpose(0,1), labels))/2

        objective = loss
        objective.backward()
        optimizer.step()

        total_objective += objective
        
    all_seq_embeds = torch.cat(all_seq_embeds, dim=0).detach().cpu().numpy()
    #print(all_keyword_embeds[-2].shape, all_keyword_embeds[-1].shape)
    all_keyword_embeds = torch.cat(all_keyword_embeds, dim=0).detach().cpu().numpy()
    ind_keyword_embeds = get_individual_keyword_embeds(net, vocab_size, device)

    return all_seq_embeds, all_keyword_embeds, ind_keyword_embeds, total_objective


def predict_clip(net, vocab_size, seqs_list, keywords_list, batch_size=64, device=None):
    total_seq_embeds = []
    total_keyword_embeds = []
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    for i in tqdm(range(0, len(seqs_list), batch_size)):
        batch_seqs = seqs_list[i:i+batch_size]

        batch_seqs = collate_padd(batch_seqs, device=device)
        batch_keywords = [torch.from_numpy(np.array(keyword_inds)).to(device) for keyword_inds in keywords_list[i:i+batch_size]]

        seq_embeds, keyword_embeds = net(batch_seqs, batch_keywords)
        seq_embeds = nn.functional.normalize(seq_embeds)

        total_seq_embeds.append(seq_embeds.detach().cpu().numpy())
        total_keyword_embeds.append(keyword_embeds.detach().cpu().numpy())

        similarity = torch.mm(seq_embeds, keyword_embeds.transpose(0,1)).squeeze()
        similarity *= torch.exp(net.temperature)
        labels = torch.arange(start=0, end=similarity.shape[0], dtype=torch.long).to(device)
        loss = (loss_fn(similarity, labels) + loss_fn(similarity.transpose(0,1), labels))/2
        total_loss += loss.item()

    total_seq_embeds = np.concatenate(total_seq_embeds) 
    total_keyword_embeds = np.concatenate(total_keyword_embeds) 

    ind_keyword_embeds = get_individual_keyword_embeds(net, vocab_size, device)

    return total_seq_embeds, total_keyword_embeds, ind_keyword_embeds, total_loss


def test_embeddings(seq_embeddings, ind_keyword_embeddings, keyword_annot_lists, all_keyword_embeddings):
    # get nearest keyword_embedding for every seq_embedding (cosine similarity)
    # if keyword_annot_lists contains that keyword embedding, treat as correct prediction
    # report accuracy out of all sequences
    sim_mat = cosine_similarity(seq_embeddings, Y=ind_keyword_embeddings) 
    #predicted_keywords = np.argmax(sim_mat, axis=1)
    k = 10
    predicted_keywords = get_top_k_element_list(sim_mat, k)
    correct = 0
    jaccard_sims = []
    for i, pred_kw_row in enumerate(predicted_keywords):
        for pred_kw in pred_kw_row:
            if pred_kw in keyword_annot_lists[i]:
                correct += 1
                break
        #pred_kw_row_set = set(pred_kw_row)
        #annot_set = set(keyword_annot_lists[i])
        #jaccard_sim = len(pred_kw_row_set.intersection(annot_set))/len(pred_kw_row_set.union(annot_set))
        #jaccard_sims.append(jaccard_sim)
    #accuracy = sum(jaccard_sims)/len(jaccard_sims)
    accuracy = correct/(seq_embeddings.shape[0])
    print(accuracy)
    assert accuracy <= 1

    seq_annot_self_sims = np.einsum('ij,ij->i', seq_embeddings, all_keyword_embeddings)
    avg_self_sim = seq_annot_self_sims.sum()/seq_annot_self_sims.shape[0]
    print('Average self sim:' + str(avg_self_sim))

    return accuracy, avg_self_sim


def compute_kmeans_nmi(feats, y_list, n_classes, test_feats=None, y_test=None):
    avg_nmi = 0
    all_feats = torch.flatten(feats, start_dim=0, end_dim=1)
    if test_feats is not None:
        all_test_feats = torch.flatten(test_feats, start_dim=0, end_dim=1)
        all_feats = torch.cat((all_feats, all_test_feats))
    km = KMeans(n_clusters=n_classes, n_jobs=None) # all processors
    print(all_feats.shape)
    non_torch_feats = all_feats.detach().cpu().numpy()
    km.fit(non_torch_feats)
    if test_feats is None:
        preds = km.predict(non_torch_feats)
        curr_nmi = nmi(preds, torch.flatten(y_list, start_dim=0, end_dim=1).detach().cpu().numpy())
    else:
        preds = km.predict(all_test_feats.detach().cpu().numpy())
        curr_nmi = nmi(preds, torch.flatten(y_test, start_dim=0, end_dim=1).detach().cpu().numpy())
    return curr_nmi


def train_clip_model(model, vocab_size, vocab, prot_list, save_prefix, train_seqs, train_keywords, epochs, batch_size, learning_rate, device):
    model.to(device)
    loss_hist = []
    print('Batch size: ' + str(batch_size))
    tensorboard_root_dir = 'seqCLIP_tensorboard_logs'
    writer = SummaryWriter(log_dir='./' + tensorboard_root_dir + '/' + save_prefix + '_lr_' + str(learning_rate) + '_batch_size_' + str(batch_size) + '/') 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print('Initial protein embeddings')
    curr_seq_embeds, curr_keyword_embeds, ind_keyword_embeds, loss = predict_clip(model, vocab_size, train_seqs, train_keywords, batch_size=batch_size, device=device)
    curr_accuracy, avg_self_sim = test_embeddings(curr_seq_embeds, ind_keyword_embeds, train_keywords, curr_keyword_embeds)
    writer.add_scalar('Epoch_Loss/contrastive_embedding_loss', loss, -1)
    writer.add_scalar('Epoch_Loss/nearest_keyword_accuracy', curr_accuracy, -1)
    writer.add_scalar('Epoch_Loss/avg_seq_annot_sim', avg_self_sim, -1)
    for epoch in range(epochs):
        net.train()
        optimizer.zero_grad()
        curr_seq_embeds, curr_keyword_embeds, ind_keyword_embeds, train_objective = train_epoch_clip(model, train_seqs, train_keywords, vocab_size, optimizer, batch_size=batch_size, device=device)
        print('Epoch ' + str(epoch) + ': Loss: ' + str(train_objective.item()))
        writer.add_scalar('Epoch_Loss/contrastive_embedding_loss', train_objective, epoch)
        loss_hist.append(train_objective.item())
        projector_embeds = np.concatenate((curr_keyword_embeds, curr_seq_embeds))
        curr_accuracy, avg_self_sim = test_embeddings(curr_seq_embeds, ind_keyword_embeds, train_keywords, curr_keyword_embeds)
        writer.add_scalar('Epoch_Loss/nearest_keyword_accuracy', curr_accuracy, epoch)
        writer.add_scalar('Epoch_Loss/avg_seq_annot_sim', avg_self_sim, epoch)
        #projector_labels = all_keywords + prot_list
        #writer.add_embedding(projector_embeds, metadata=projector_labels, global_step=epoch, tag='curr_embeds')
        print(model.temperature)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            torch.save(net.state_dict(), './saved_models/' + save_prefix + '_epoch_' + str(epoch) + '.pckl')
    writer.close()
    trained_prot_embeds, curr_keyword_embeds, trained_individual_keyword_embeds, loss = predict_clip(model, vocab_size, train_seqs, train_keywords, batch_size=batch_size, device=device)
    return trained_prot_embeds, trained_individual_keyword_embeds, loss_hist


def arguments():
    args = argparse.ArgumentParser()
    args.add_argument('--learning_rate', type=float, default=0.01)
    args.add_argument('--epochs', type=int, default=200)
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--embed_dim', type=int, default=100)
    args.add_argument('--alpha', type=float, default=0.0)
    args.add_argument('--save_prefix', type=str, default='no_save_prefix')
    args.add_argument('--fasta_fname', type=str)
    args.add_argument('--load_model', type=str, default=None, help='load model to continue training')
    args.add_argument('--load_model_predict', type=str, default=None, help='load model to predict only')
    args.add_argument('--keyword_file', type=str, default=None)
    args.add_argument('--brenda', type=str, default=None)
    args.add_argument('--go_term_list_file', type=str, default=None)

    args = args.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    device = torch.device('cuda')
    np.random.seed(seed=RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    args = arguments()

    #training_features_1, training_features_2, test_features, train_1_graph, train_2_graph, test_graph, y_train_1, y_train_2, y_test, R_12, R_13, R_23 = make_dataset(3*num_nodes, feat_dim, feat_dim, num_classes, prop_known_sims=args.prop_known_sims, device=device)

    #train_feats, train_graphs, test_feats, test_graphs, train_intergraph_conns, train_test_intergraph_conns, test_intergraph_conns, train_y_list, test_y_list = make_multi_graph_dataset(n_graphs, num_nodes, feat_dim, feat_dim, num_classes, prop_known_sims=args.prop_known_sims, device=device)

    alpha = args.alpha

    id2seq = load_fasta(args.fasta_fname)
    
    #one_hot_seqs = np.array([seq2onehot(id2seq[prot]) for prot in id2seq.keys()])
    keyword_dict = pickle.load(open(args.keyword_file, 'rb'))
    keyword_df = keyword_dict['keyword_df']
    train_keywords = np.array(keyword_df['keyword_inds'])
    train_seqs = np.array([seq2onehot(id2seq[prot]) for prot in keyword_df['Entry']])
    all_keywords = keyword_dict['all_keywords']
    prot_list = keyword_df['Entry'].tolist()

    print('First sequence one-hot shape')
    print(train_seqs[0].shape)
    seq_dim = train_seqs[0].shape[1]
    print('First keyword indices')
    print(train_keywords[0])
    keyword_vocab_size = len(all_keywords)
    print(keyword_vocab_size)
    embed_dim = args.embed_dim


    save_prefix = args.save_prefix
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    if args.load_model is not None:
        #net = Conv_NN(seq_dim, num_classes, hidden_dim=args.hidden)
        net = seqCLIP(seq_dim, keyword_vocab_size, embed_dim)
        state_dict = torch.load(args.load_model)
        net.load_state_dict(state_dict)
        net.to(device)
        print('Continuing to train model for ' + str(args.epochs)) 
        trained_seq_embeds, trained_individual_keyword_embeds, loss_hist = train_clip_model(net, keyword_vocab_size, all_keywords, prot_list, save_prefix, train_seqs, train_keywords, epochs, batch_size, learning_rate, device)
        nets['loss_history'] = [loss for loss in loss_hist]
    elif args.load_model_predict is not None:
        net = seqCLIP(seq_dim, keyword_vocab_size, embed_dim)
        state_dict = torch.load(args.load_model_predict)
        net.load_state_dict(state_dict)
        net.to(device)
        total_seq_embeds, _, ind_keyword_embeds, total_loss = predict_clip(net, vocab_size, train_seqs, train_keywords, batch_size=batch_size, device=device)
        trained_individual_keyword_embeds = get_individual_keyword_embeds(model, vocab_size, device)
    else:
        net = seqCLIP(seq_dim, keyword_vocab_size, embed_dim)
        trained_seq_embeds, trained_individual_keyword_embeds, loss_hist = train_clip_model(net, keyword_vocab_size, all_keywords, prot_list, save_prefix, train_seqs, train_keywords, epochs, batch_size, learning_rate, device)

    outputs = {}
    outputs['prots'] = np.array(keyword_df['Entry'])
    outputs['loss_history'] = [loss for loss in loss_hist]
    outputs['trained_seq_embeddings'] = trained_seq_embeds
    outputs['trained_keyword_embeddings'] = trained_individual_keyword_embeds
    outputs['all_keywords'] = all_keywords


    pickle.dump(outputs, open(save_prefix + '_trained_embeddings.pckl', 'wb'))
    print('Saved trained features')
