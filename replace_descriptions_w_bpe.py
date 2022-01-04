import pandas as pd
import sys
import csv

input_fname = sys.argv[1]
codified_desc_fname = sys.argv[2]
#out_fname = '.'.join(input_fname.split('.')[:-1]) + '_codified.tsv'
out_fname = sys.argv[3]

train = pd.read_csv(input_fname, sep='\t')
codified_descs = open(codified_desc_fname, 'r').read().split('\n')

train['GO-def'] = codified_descs # replace with codified versions of descriptions
train.to_csv(out_fname, sep='\t', index=False, quoting=csv.QUOTE_NONE, escapechar=' ')
