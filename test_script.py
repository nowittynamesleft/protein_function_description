from data import SequenceGOCSVDataset, seq_go_collate_pad
from torch.utils.data import DataLoader, Subset
import torch
#from models import NMTDescriptionGen
from alt_transformer_model import SeqSet2SeqTransformer, create_mask
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.plugins import DDPPlugin # for find_unused_parameters=False; this is True by default which gives a performance hit, and according to documentation
import argparse
import numpy as np

def arguments():
    args = argparse.ArgumentParser()
    #args.add_argument('--learning_rate', type=float, default=0.01)
    args.add_argument('annot_seq_file', type=str)
    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--batch_size', type=int, default=1)
    args.add_argument('--seq_set_len', type=int, default=32)
    args.add_argument('--emb_size', type=int, default=256)
    args.add_argument('--save_prefix', type=str, default='no_save_prefix')
    args.add_argument('--fasta_fname', type=str)
    args.add_argument('--load_model', type=str, default=None, help='load model to continue training')
    args.add_argument('--load_model_predict', type=str, default=None, help='load model to predict only')

    args = args.parse_args()
    print(args)
    return args


class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.metrics.append(trainer.logged_metrics['loss'])

def convert_preds_to_words(predictions, vocab):
    word_preds = []
    for batch in predictions:
        word_preds.append([])
        for sample in batch:
            word_preds[-1].append([vocab[ind] for ind in sample])
    return word_preds


#device = torch.device("cuda:0")
args = arguments()
seq_set_len = args.seq_set_len
emb_size = args.emb_size
#x = SequenceGODataset('first_6_prots.fasta', 'fake_data.pckl', seq_set_len, device=device)
#x = SequenceGOCSVDataset('uniprot_sprot_training_annot.csv', seq_set_len, device=device)
x = SequenceGOCSVDataset(args.annot_seq_file, seq_set_len)
#x = SequenceGOCSVDataset('uniprot_sprot_test_annot.csv', seq_set_len, device=device)
#import ipdb; ipdb.set_trace()

dl_workers = 4
#num_gpus = 4
num_gpus = 1
#dl_workers = 1
#dl_workers = 0

dl = DataLoader(x, batch_size=args.batch_size, collate_fn=x.collate_fn, num_workers=dl_workers, pin_memory=True)

batch = next(iter(dl))

S_padded, S_pad_mask, GO_padded, GO_pad_mask = batch 

#model = NMTDescriptionGen(len(x.alphabet), len(x.vocab), 80, num_heads=1).to(device)
#model = SeqSet2SeqTransformer(num_encoder_layers=1, num_decoder_layers=1, 
#        emb_size=emb_size, src_vocab_size=len(x.alphabet), tgt_vocab_size=len(x.vocab), 
#       dim_feedforward=512, num_heads=4, dropout=0.0, vocab=x.vocab).to(device)
model = SeqSet2SeqTransformer(num_encoder_layers=1, num_decoder_layers=1, 
        emb_size=emb_size, src_vocab_size=len(x.alphabet), tgt_vocab_size=len(x.vocab), 
       dim_feedforward=512, num_heads=4, dropout=0.0, vocab=x.vocab)

# what kind of masks are these?
print(S_padded.shape)
print(S_pad_mask.shape)
print(GO_padded.shape)
#src_mask, tgt_mask = create_mask(S_padded, GO_padded)
# i think the source mask should not hide anything from the source
# memory_key_padding_mask is optional
'''
tgt_mask = 
src_padding_mask = 
tgt_padding_mask = 
#memory_key_padding_mask =
print('Pad masks')
print('S_pad_mask')
print(S_pad_mask.shape)
print('GO_pad_mask')
print(GO_pad_mask.shape)
print('src_mask')
print(src_mask.shape)
print('tgt_mask')
print(tgt_mask.shape)

print(tgt_mask.device)
print(S_pad_mask.device)
print(GO_pad_mask.device)
print(src_mask.device)
'''
print('Vocab size:')
print(len(x.vocab))

#output = model(src=S_padded.to(device), trg=GO_padded.to(device), src_mask=src_mask.to(device), 
#               tgt_mask=tgt_mask.to(device), src_padding_mask=S_pad_mask.to(device),
#               tgt_padding_mask=GO_pad_mask, memory_key_padding_mask=None)
#print(output)

metric_callback = MetricsCallback()
#trainer = Trainer(gpus=1, max_epochs=20, callbacks=metric_callback)
trainer = Trainer(gpus=num_gpus, max_epochs=args.epochs, auto_select_gpus=True, 
        callbacks=metric_callback, strategy=DDPPlugin(find_unused_parameters=False))
#import ipdb; ipdb.set_trace()
if args.load_model_predict is None:
    if args.load_model is not None:
        print('Loading model for training: ' + args.load_model)
        ckpt = torch.load(args.load_model)
        model.load_state_dict(ckpt['state_dict'])
    trainer.fit(model, dl)
    logged_metrics = metric_callback.metrics
#import ipdb; ipdb.set_trace()
    print('Logged_metrics')
    print(logged_metrics)
#trained_seq_embeds, trained_individual_keyword_embeds = trainer.predict(model, seq_kw_dataloader)
else:
    print('Loading model for predicting only: ' + args.load_model_predict)
    ckpt = torch.load(args.load_model_predict)
    model.load_state_dict(ckpt['state_dict'])

subset = Subset(x, list(range(100)))
test_dl = DataLoader(subset, batch_size=args.batch_size, collate_fn=x.collate_fn, num_workers=dl_workers, pin_memory=True)
predictions = trainer.predict(model, test_dl)
#print('Predictions')
#print(predictions) # this is num_batches x num_samples_per_batch x num_seqs_per_set_sample x max_desc_len, need to be num_batches x num_samples_per_batch x max_desc_len
word_preds = convert_preds_to_words(predictions, x.vocab)
#acc = trainer.test(model, test_dl, verbose=True)[0]['acc']
#print('Exact match accuracy')
#print(acc)
            
print('Batch preds:')
print(word_preds[0])
print('Actual descriptions:')
print(x.go_descriptions[:args.batch_size]) # assumes subset is from the first N samples of the training set

outfile = open(args.save_prefix + 'first_100_preds.txt', 'w')
import ipdb; ipdb.set_trace()
for i in range(100):
    outfile.write('Prediction:\n' + ' '.join(word_preds[int(i/args.batch_size)][i%args.batch_size]) + '\nActual description:\n' + ' '.join(x.go_descriptions[i]) + '\n')
outfile.close()
