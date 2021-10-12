from data import SequenceGODataset, seq_go_collate_pad
from torch.utils.data import DataLoader
from models import NMTDescriptionGen

x = SequenceGODataset('first_6_prots.fasta', 'fake_data.pckl', 2)

dl = DataLoader(x, batch_size=2, collate_fn=x.collate_fn)

batch = next(iter(dl))

S_padded, S_mask, batch_go_descs = batch 

nmt = NMTDescriptionGen(len(x.alphabet), len(x.vocab), 80, num_heads=1)
output = nmt(S_padded, S_mask)
print(output)
