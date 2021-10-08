from data import SequenceGODataset, seq_go_collate_pad
from torch.utils.data import DataLoader

x = SequenceGODataset('first_6_prots.fasta', 'fake_data.pckl', 2)

dl = DataLoader(x, batch_size=2, collate_fn=x.collate_fn)

batch = next(iter(dl))

S_padded, batch_go_descs = batch 
print(S_padded)
print("GO descriptions")
print(batch_go_descs)


