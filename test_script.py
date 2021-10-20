from data import SequenceGODataset, seq_go_collate_pad
from torch.utils.data import DataLoader
import torch
#from models import NMTDescriptionGen
from alt_transformer_model import SeqSet2SeqTransformer

x = SequenceGODataset('first_6_prots.fasta', 'fake_data.pckl', 2)
device = torch.device("cuda:0")

dl = DataLoader(x, batch_size=2, collate_fn=x.collate_fn)

batch = next(iter(dl))

S_padded, S_mask, batch_go_descs = batch 

#nmt = NMTDescriptionGen(len(x.alphabet), len(x.vocab), 80, num_heads=1).to(device)
nmt = SeqSet2SeqTransformer(num_encoder_layers=1, num_decoder_layers=1, emb_size=256, src_vocab_size=len(x.alphabet), tgt_vocab_size=len(x.vocab), dim_feedforward=512, num_heads=4, dropout=0.0).to(device)

# what kind of masks are these?
'''
tgt_mask = 
src_padding_mask = 
tgt_padding_mask = 
memory_key_padding_mask =
'''

output = nmt(src=S_padded.to(device), trg=batch_go_descs, src_mask=S_mask.to(device))
print(output)
