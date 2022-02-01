from data import SequenceGOCSVDataset, seq_go_collate_pad, SequenceDataset
from torch.utils.data import DataLoader, Subset
import torch
#from models import NMTDescriptionGen
from alt_transformer_model import SeqSet2SeqTransformer, create_mask
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.plugins import DDPPlugin # for find_unused_parameters=False; this is True by default which gives a performance hit, and according to documentation
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse
import numpy as np
import pickle
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tqdm
from utils import accuracy, micro_aupr

obofile = 'go.obo'


def arguments():
    args = argparse.ArgumentParser()
    #args.add_argument('--learning_rate', type=float, default=0.01)
    args.add_argument('annot_seq_file', type=str)
    args.add_argument('--test_annot_seq_file', type=str, default=None)
    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--batch_size', type=int, default=1)
    args.add_argument('--seq_set_len', type=int, default=32)
    args.add_argument('--min_tf_prob', type=float, default=1.0, 
            help='Minimum teacher forcing prob (for scheduled sampling). Default is 1.0, so all teacher forcing.')
    args.add_argument('--emb_size', type=int, default=256)
    args.add_argument('--save_prefix', type=str, default='no_save_prefix')
    args.add_argument('--fasta_fname', type=str)
    args.add_argument('--load_model', type=str, default=None, 
            help='load model to continue training')
    args.add_argument('--load_model_predict', type=str, default=None, 
            help='load model to predict only')
    args.add_argument('--num_pred_terms', type=int, default=-1, 
            help='how many descriptions to predict to compare to real go terms')
    args.add_argument('--test', action='store_true', 
            help='code testing flag, do not save model checkpoints, only train on num_pred_terms GO terms')
    args.add_argument('--classify', action='store_true', 
            help='Classify the test dataset into the available GO terms of the test dataset')
    args.add_argument('--generate', action='store_true', 
            help='Generate descriptions for the test dataset protein sets')
    args.add_argument('--load_vocab', type=str, default=None, 
            help='Load vocab from pickle file instead of assuming all description vocab is included in annot_seq_file')

    args = args.parse_args()
    print(args)
    return args


class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = {}
        self.metrics['loss'] = []
        self.metrics['val_loss'] = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.metrics['loss'].append(trainer.logged_metrics['loss_epoch'])
    
    def on_validation_epoch_end(self, trainer, pl_module):
        self.metrics['val_loss'].append(trainer.logged_metrics['val_loss_epoch'])

def convert_preds_to_words(predictions, vocab):
    # converts all predictions in a set of batches to words in a num_batches-length list of batch_size-length lists of sentence-length lists of strings
    word_preds = []
    for batch in predictions:
        word_preds.append([])
        for sample in batch:
            word_preds[-1].append([vocab[ind] for ind in sample])
    return word_preds


def convert_sample_preds_to_words(predictions, vocab):
    # converts all predictions in a set of batches to words in a num_batches*batch_size-length list of sentence-length lists of strings
    word_preds = []
    for batch in predictions:
        for sample in batch:
            word_preds.append([vocab[ind] for ind in sample])
    return word_preds


def convert_single_batch_preds_to_words(predictions, vocab):
    # converts all predictions in a set of batches to words in a num_batches*batch_size-length list of sentence-length lists of strings
    word_preds = []
    for sample in predictions:
        word_preds.append([vocab[ind] for ind in sample])
    return word_preds


def get_prot_preds(fasta_file, trainer, model, combined=False):
    seq_dataset = SequenceDataset(fasta_file)
    if not combined:
        # get invididual protein preds
        dl = DataLoader(seq_dataset, batch_size=32, collate_fn=partial(seq_go_collate_pad, seq_set_size=1), num_workers=dl_workers, pin_memory=True)
        preds = trainer.predict(model, dl)
    else:
        print('Producing a single combined prediction for all proteins in the following fasta file: ' + fasta_file)
        seqs = seq_dataset.seqs
        pred_batch = seq_go_collate_pad([(seqs,)], seq_set_size=len(seqs)) # make into a list to make it a "batch" of one
        preds = model.predict_step(pred_batch, 0)
    return seq_dataset.prot_list, preds


def get_individual_go_term_dataloaders(dataset, num_to_predict, max_seq_set_size=128):
    dls = []
    ground_truths = []
    included_go_term_inds = []
    for go_term_ind in range(num_to_predict):
        annotated_seqs = dataset.get_annotated_seqs(go_term_ind)
        if len(annotated_seqs[0]) < max_seq_set_size: # only if you can actually compute the annotated sequences in the model
            print(str(go_term_ind) + ' out of ' + str(num_to_predict))
            print('Number of annotated seqs:' + str(len(annotated_seqs[0])))
            
            curr_collate_fn = partial(seq_go_collate_pad, seq_set_size=len(annotated_seqs[0]))
            curr_pred_dl = get_subset_dataloader(dataset, [go_term_ind], batch_size=1, collate_fn=curr_collate_fn)
            dls.append(curr_pred_dl)
            ground_truth_desc = dataset.go_descriptions[go_term_ind]
            ground_truths.append(ground_truth_desc)
            included_go_term_inds.append(go_term_ind)
    return dls, ground_truths, included_go_term_inds


def predict_all_prots_of_go_term(trainer, model, num_pred_terms, save_prefix, dataset, evaluate_probs=False):
    # generate predictions for all proteins of a given GO term, for all GO terms in the dataset given as the argument
    preds = []
    ground_truth_descs = []
    dataset.set_include_go_mode(evaluate_probs) # do not send the predict_step function the actual GO descriptions if not evaluating probs
    dataset.set_sample_mode(False) # do not sample proteins for a particular go term; use all of them
    print('One GO term full set')
    if num_pred_terms == -1 or num_pred_terms == len(dataset):
        num_to_predict = len(dataset)
    else:
        num_to_predict = num_pred_terms
    print('Number to predict: ' + str(num_to_predict))
    dls, ground_truths, included_go_term_inds = get_individual_go_term_dataloaders(dataset, num_to_predict)

    if evaluate_probs:
        model.pred_pair_probs = True
        all_desc_log_probs = trainer.predict(model, dls)
        all_desc_log_probs = [log_prob[0] for log_prob in all_desc_log_probs]
        all_desc_probs = torch.exp(torch.cat(all_desc_log_probs))
        print('Average probability of true GO descriptions:')
        average_true_desc_prob = sum(all_desc_probs)/len(all_desc_probs)
        print(average_true_desc_prob)
        outfile = open(save_prefix + '_full_seq_sets_pair_probabilities.txt', 'w')
        for i, ground_truth_desc in enumerate(ground_truths):
            outfile.write('Ground truth description: ' + ground_truth_desc + '\nProbability assigned: ' + str(all_desc_probs[i].item()) + '\n\n')
        outfile.write('Average probability score of true GO descriptions:\n' + str(average_true_desc_prob.item()))
        outfile.close()
        return average_true_desc_prob

    model.pred_pair_probs = False
    output = trainer.predict(model, dls)
    preds = [pred[0][0][0] for pred in output]
    probs = [pred[0][1][0] for pred in output]
    word_preds = convert_preds_to_words(preds, dataset.vocab) # will be a num_pred_terms-length list of beam_width-length lists of sentence-length lists of words

    if num_pred_terms == -1 or num_pred_terms == len(dataset):
        outfile = open(save_prefix + '_full_seq_sets_all_preds.txt', 'w')
    else:
        outfile = open(save_prefix + '_full_seq_sets_first_' + str(num_pred_terms) + '_preds.txt', 'w')
    for i, word_pred in enumerate(word_preds):
        outfile.write('Prediction:\n' + ' '.join(word_pred[0]) + '\nActual description:\n' + ground_truths[i] + '\n\n')
    outfile.close()


def predict_subsample_prots_go_term_descs(trainer, model, test_dl, test_dataset, save_prefix):
    pred_output = trainer.predict(model, test_dl)
    
    preds = [pred for batch in pred_output for pred in batch[0]]
    probs = [prob for batch in pred_output for prob in batch[1]]
    word_preds = convert_sample_preds_to_words(preds, model.vocab)
    outfile = open(save_prefix + '_subsample_prot_preds.txt', 'w')
    for pred in range(preds):
        included_term_ind = included_term_inds[i]
        outfile.write('GO term: ' + test_dataset.go_terms[included_term_ind] + ': ' + test_dataset.go_names[included_term_ind] + '\n')
        outfile.write('Predictions:\n')
        outfile.write(pred)
        outfile.write('Actual description:\n' + ' '.join(x.go_descriptions[included_term_ind]) + '\n\n')
    outfile.close()
    

def all_combined_fasta_description(model, trainer, fasta_fname, vocab, save_prefix):
    # generate predictions for all proteins in a fasta
    prot_ids, prot_preds = get_prot_preds(fasta_fname, trainer, model, combined=True)
    word_preds = convert_sample_preds_to_words([prot_preds], vocab)
    assert len(word_preds) == 1
    word_preds = word_preds[0]
    outfile = open(save_prefix + '_all_prot_preds.txt', 'w')
    outfile.write('Proteins: ' + ','.join(prot_ids) + '\nPrediction:\n' + ' '.join(word_preds) + '\n')
    outfile.close()

    
def one_by_one_prot_fasta_description(model, fasta_fname, trainer, seq_dataset, save_prefix):
    # generate predictions one by one for each protein in a fasta
    prot_ids, prot_preds = get_prot_preds(fasta_fname, trainer, model)
    word_preds = convert_sample_preds_to_words(prot_preds, seq_dataset.vocab)
    outfile = open(save_prefix + '_single_prot_preds.txt', 'w')
    for i in range(len(word_preds)):
        outfile.write('Protein: ' + prot_ids[i] + '\nPrediction:\n' + ' '.join(word_preds[i]) + '\n')
    outfile.close()


def single_prot_description(model, annot_seq_file, loaded_vocab, save_prefix, num_pred_terms):
    # generate predictions for each protein one by one in the first num_pred_terms GO terms
    x = SequenceGOCSVDataset(annot_seq_file, obofile, 1, vocab=loaded_vocab)
    total_word_preds = []
    go_term_ind = 0
    included_term_inds = []
    while len(included_term_inds) < num_pred_terms:
        annotated_seqs = x.get_annotated_seqs(go_term_ind)
        if len(annotated_seqs[0]) < 50: # only take very specific terms to compare to (these are single proteins, so should compare specific functions)
            annotated_seqs = annotated_seqs[0][:5]
            predictions = []
            for sample_prot_ind in range(5):
                annotated_seq = [annotated_seqs[sample_prot_ind]]
                pred_batch = seq_go_collate_pad([(annotated_seq,)], seq_set_size=1) # make into a list to make it a "batch" of one
                predictions.append(model.predict_step(pred_batch, go_term_ind))
            curr_go_term_preds = convert_sample_preds_to_words(predictions, x.vocab)
            total_word_preds.append(curr_go_term_preds)
            included_term_inds.append(go_term_ind)
        go_term_ind += 1

    outfile = open(save_prefix + 'single_prot_first_' + str(num_pred_terms) + '_GO_term_preds.txt', 'w')
    for i in range(num_pred_terms):
        included_term_ind = included_term_inds[i]
        outfile.write('GO term: ' + x.go_terms[included_term_ind] + ': ' + x.go_names[included_term_ind] + '\n')
        outfile.write('Predictions:\n')
        for j in range(5): # 5 random prots for each
            outfile.write(' '.join(total_word_preds[i][j]) + '\n')
        outfile.write('Actual description:\n' + ' '.join(x.go_descriptions[included_term_ind]) + '\n\n')
    outfile.close()


def classification(model, dataset, save_prefix='no_prefix', subsample=False):
    # extract each seq set, compute all pairs of probabilities
    GO_padded, GO_pad_masks = dataset.get_padded_descs()
    dataset.set_include_go_mode(False)
    if subsample:
        included_go_inds = np.arange(len(dataset))
    else:
        dataloaders, _, included_go_inds = get_individual_go_term_dataloaders(dataset, len(dataset), max_seq_set_size=128) # do somewhat specific ones just to take less time...
    preds = []
    all_pred_token_probs = []
    # no len penalty
    preds_2 = []
    all_pred_token_probs_2 = []
    # tqdm progress bar
    print(str(len(included_go_inds)) + "-way zero-shot classification.")
    if subsample:
        print("Subsampling the sequence sets to make predictions")
    for ind in tqdm.tqdm(included_go_inds): # only take the indices of terms that have few enough sequences annotated (to fit in gpu memory)
        if subsample:
            seq_set = dataset[ind]
        else:
            seq_set = dataset.get_annotated_seqs(ind)
        S_padded, S_mask = seq_go_collate_pad([seq_set], seq_set_size=len(seq_set[0]))
        #S_padded = S_padded.to(model.device)
        #S_mask = S_mask.to(model.device)
        seq_set_desc_probs, seq_set_desc_token_probs = model.classify_seq_set(S_padded, S_mask, GO_padded, GO_pad_masks, len_penalty=True) # batch sizes of 1 each, index out of it
        seq_set_desc_probs_2, seq_set_desc_token_probs_2 = model.classify_seq_set(S_padded, S_mask, GO_padded, GO_pad_masks, len_penalty=False) # batch sizes of 1 each, index out of it
        preds.append(seq_set_desc_probs)
        all_pred_token_probs.append(seq_set_desc_token_probs)
        #preds.append(seq_set_desc_probs)
        preds_2.append(seq_set_desc_probs_2)
        all_pred_token_probs_2.append(seq_set_desc_token_probs_2)

    #acc = accuracy_score(preds, included_go_inds)
    #import ipdb; ipdb.set_trace()
    #preds = torch.tensor(preds).transpose(0,1)
    preds = torch.tensor(preds)
    included_go_inds = torch.tensor(included_go_inds)
    preds_2 = torch.tensor(preds_2)
    pred_outdict = {'seq_set_go_term_inds': included_go_inds, 'all_term_preds': preds, 'all_term_token_probs': seq_set_desc_token_probs, 'go_descriptions': np.array(dataset.go_descriptions)}
    pickle.dump(pred_outdict, open(save_prefix + '_pred_dict.pckl', 'wb'))

    if len(preds) < 1000:
        accs = accuracy(preds, included_go_inds, topk=(5, 4, 3, 2, 1))
        print('Len penalty: Top 5, 4, 3, 2, 1 accuracies: ' + str(accs))
        accs_2 = accuracy(preds_2, included_go_inds, topk=(5, 4, 3, 2, 1))
        print('No len penalty: Top 5, 4, 3, 2, 1 accuracies: ' + str(accs_2))
    else:
        accs = accuracy(preds, included_go_inds, topk=(1000, 500, 100, 50, 10, 5, 1))
        print('Len penalty Top 1000, 500, 100, 50, 10, 5, 1 accuracies: ' + str(accs))
        accs_2 = accuracy(preds_2, included_go_inds, topk=(1000, 500, 100, 50, 10, 5, 1))
        print('No len penalty Top 1000, 500, 100, 50, 10, 5, 1 accuracies: ' + str(accs_2))

    #aupr = micro_aupr(preds, dataset.go_annot_mat)

    return preds, accs
    

def get_subset_dataloader(dataset, inds, batch_size, collate_fn):
    data = Subset(dataset, inds)
    dl = DataLoader(data, batch_size=batch_size, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    return dl


def get_train_val_dataloaders(full_dataset, batch_size, collate_fn, test=False):
    num_samples = len(full_dataset)
    if test:
        num_samples = 10
    train_inds, val_inds = train_test_split(range(0, num_samples))
    train_dl = get_subset_dataloader(full_dataset, train_inds, batch_size, collate_fn)
    val_dl = get_subset_dataloader(full_dataset, val_inds, batch_size, collate_fn)
    return train_dl, val_dl


if __name__ == '__main__':
    args = arguments()
    np.random.seed(973)
    seq_set_len = args.seq_set_len
    emb_size = args.emb_size
    if args.load_vocab is not None:
        loaded_vocab = pickle.load(open(args.load_vocab, 'rb'))
        x = SequenceGOCSVDataset(args.annot_seq_file, obofile, seq_set_len, vocab=loaded_vocab, save_prefix=args.save_prefix)
    else:
        vocab_fname = args.save_prefix + '_vocab.pckl'
        x = SequenceGOCSVDataset(args.annot_seq_file, obofile, seq_set_len, save_prefix=args.save_prefix)
        pickle.dump(x.vocab, open(vocab_fname, 'wb'))

    #num_gpus = torch.cuda.device_count()
    num_gpus = 1
    dl_workers = 0
    #dl_workers = num_gpus # one per gpu I guess?
#num_gpus = 1
#dl_workers = 0
    num_pred_terms = args.num_pred_terms
    if args.num_pred_terms == -1:
        num_pred_terms = len(x)


    print('Vocab size:')
    print(len(x.vocab))
    collate_fn = x.collate_fn
    model = SeqSet2SeqTransformer(num_encoder_layers=1, num_decoder_layers=1, 
            emb_size=emb_size, src_vocab_size=len(x.alphabet), tgt_vocab_size=len(x.vocab), 
            dim_feedforward=512, num_heads=4, dropout=0.0, vocab=x.vocab, 
            min_tf_prob=args.min_tf_prob)

    if args.test_annot_seq_file is None:
        test_dataset = Subset(x, list(range(num_pred_terms)))
        test_dl = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=dl_workers, pin_memory=True)
    else:
        test_dataset = SequenceGOCSVDataset(args.test_annot_seq_file, obofile, seq_set_len, vocab=x.vocab)
        test_dl = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=dl_workers, pin_memory=True)

    #metric_callback = MetricsCallback()
    early_stopping_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10)
    csv_logger = CSVLogger('lightning_logs', name=(args.save_prefix + '_experiment'))
    #trainer = Trainer(gpus=num_gpus, max_epochs=args.epochs, auto_select_gpus=True,  # mixed precision causes nan loss, so back to regular precision.
            #callbacks=metric_callback, strategy=DDPPlugin(find_unused_parameters=False), checkpoint_callback=(not args.test), logger=(not args.test))
    #trainer = Trainer(gpus=num_gpus, max_epochs=args.epochs,  # mixed precision causes nan loss, so back to regular precision.
    #        callbacks=[metric_callback, early_stopping_callback], logger=csv_logger)
    #trainer = Trainer(gpus=num_gpus, max_epochs=args.epochs,  # mixed precision causes nan loss, so back to regular precision.
    #        callbacks=[metric_callback], logger=csv_logger)
    trainer = Trainer(gpus=num_gpus, max_epochs=args.epochs,  # mixed precision causes nan loss, so back to regular precision.
            callbacks=[early_stopping_callback], logger=csv_logger)

    if args.load_model_predict is None:
        if args.load_model is not None:
            print('Loading model for training: ' + args.load_model)
            ckpt = torch.load(args.load_model)
            model.load_state_dict(ckpt['state_dict'])
            model.to('cuda:0')
            print('Classfication before training:')
            preds, acc = classification(model, test_dataset, save_prefix=args.save_prefix, subsample=True)
        train_dl, val_dl = get_train_val_dataloaders(x, args.batch_size, collate_fn, test=args.test)
        print('Validation before training...')
        trainer.validate(model, val_dl)
        print('Training...')
        trainer.fit(model, train_dl, val_dl)
        #dataloader = DataLoader(x, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=dl_workers, pin_memory=True)
        #trainer.fit(model, dataloader)
        #logged_metrics = metric_callback.metrics
        #print('Logged_metrics')
        #print(logged_metrics)
    else:
        print('Loading model for predicting only: ' + args.load_model_predict)
        ckpt = torch.load(args.load_model_predict)
        model.load_state_dict(ckpt['state_dict'])

    print('Teacher forcing probability after training:')
    print(model.tf_prob)
    #average_true_desc_prob = predict_all_prots_of_go_term(trainer, model, num_pred_terms, args.save_prefix, x, evaluate_probs=True)
    model.to('cuda:0')
    if args.classify:
        print('Classfication after whole script:')
        preds, acc = classification(model, test_dataset, save_prefix=args.save_prefix, subsample=True)
    if args.generate:
        print('Generation after whole script:')
        if model.pred_pair_probs:
            model.pred_pair_probs = False
        trainer.predict(model, test_dl)

    #print('Predict test:')
    #model.pred_pair_probs = True
    #trainer.predict(model, test_dl)
    #predict_subsample_prots_go_term_descs(trainer, model, test_dl, test_dataset, args.save_prefix)
    