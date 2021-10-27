from data import SequenceGODataset, seq_go_collate_pad
from torch.utils.data import DataLoader
import torch
#from models import NMTDescriptionGen
from alt_transformer_model import SeqSet2SeqTransformer, create_mask
from pytorch_lightning import Trainer, Callback

class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


device = torch.device("cuda:0")
x = SequenceGODataset('first_6_prots.fasta', 'fake_data.pckl', 2, device=device)

dl = DataLoader(x, batch_size=2, collate_fn=x.collate_fn)

batch = next(iter(dl))

S_padded, S_pad_mask, GO_padded, GO_pad_mask = batch 

#model = NMTDescriptionGen(len(x.alphabet), len(x.vocab), 80, num_heads=1).to(device)
model = SeqSet2SeqTransformer(num_encoder_layers=1, num_decoder_layers=1, 
        emb_size=256, src_vocab_size=len(x.alphabet), tgt_vocab_size=len(x.vocab), 
        dim_feedforward=512, num_heads=4, dropout=0.0).to(device)

# what kind of masks are these?
print(S_padded.shape)
print(S_pad_mask.shape)
print(GO_padded.shape)
src_mask, tgt_mask = create_mask(S_padded, GO_padded, device=device)
# i think the source mask should not hide anything from the source
# memory_key_padding_mask is optional
'''
tgt_mask = 
src_padding_mask = 
tgt_padding_mask = 
#memory_key_padding_mask =
'''
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
print('Vocab size:')
print(len(x.vocab))

print('Max index of GO_padded')
print(torch.max(GO_padded))
output = model(src=S_padded.to(device), trg=GO_padded.to(device), src_mask=src_mask.to(device), 
        tgt_mask=tgt_mask.to(device), src_padding_mask=S_pad_mask.to(device),
        tgt_padding_mask=GO_pad_mask, memory_key_padding_mask=None)
print(output)

metric_callback = MetricsCallback()
trainer = Trainer(gpus=1, max_epochs=10, callbacks=metric_callback)
trainer.fit(model, dl)
logged_metrics = metric_callback.metrics
print(logged_metrics)
#trained_seq_embeds, trained_individual_keyword_embeds = trainer.predict(model, seq_kw_dataloader)
predictions = trainer.predict(model, dl)[0]
print(predictions)
