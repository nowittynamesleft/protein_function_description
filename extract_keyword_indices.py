import pandas as pd
import sys
import pickle

keyword_csv = sys.argv[1]
keyword_df = pd.read_csv(keyword_csv, sep='\t')
print('Before nan removal: ' + str(keyword_df.shape))
# first remove nan rows:
keyword_df.dropna(inplace=True)
print('After nan removal: ' + str(keyword_df.shape))

entries = keyword_df['Entry']
unsep_keyword_lists = keyword_df['Keywords'].tolist()


all_keywords = []
keyword_lists = []
for i, unsep_keyword_list in enumerate(unsep_keyword_lists):
    keyword_list = unsep_keyword_list.split(';')
    keyword_lists.append(keyword_list)
    all_keywords.extend(keyword_list)

all_keywords = list(set(all_keywords))
keyword2ind = {keyword: all_keywords.index(keyword) for keyword in all_keywords}
print('Vocab size: ' + str(len(all_keywords)))

keyword_inds = []
for keyword_list in keyword_lists:
    keyword_inds.append([keyword2ind[keyword] for keyword in keyword_list])

keyword_df['keyword_inds'] = keyword_inds
pickle_dict = {'all_keywords': all_keywords, 'keyword_df': keyword_df}
pickle.dump(pickle_dict, open('.'.join(keyword_csv.split('.')[:-1]) + '_with_keyword_inds.pckl','wb'))

