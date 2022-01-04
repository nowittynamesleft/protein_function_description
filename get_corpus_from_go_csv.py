import pandas as pd
import sys
from torchtext.data.utils import get_tokenizer

input_fname = sys.argv[1]
output_fname = sys.argv[2]

tokenizer = get_tokenizer('basic_english')
train = pd.read_csv(input_fname, sep='\t')
corpus = train['GO-def']
tokenized = [' '.join(tokenizer(desc)) for desc in corpus]
outfile = open(output_fname, 'w')
outfile.write('\n'.join(tokenized))
outfile.close()
