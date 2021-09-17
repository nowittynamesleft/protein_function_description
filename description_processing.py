import pandas as pd
import pickle
import sys
from sklearn.feature_extraction.text import CountVectorizer

def main():
    fname = sys.argv[1]
    print('Processing ' + str(fname))
    df = pd.read_csv(fname, sep='\t')
    print('removing nans')
    df = df.dropna()
    print(df['Function [CC]'])
    #all_strings = ' '.join(df['Function [CC]'])
    #print(all_strings)
    cv = CountVectorizer()
    cv.fit_transform(df['Function [CC]'])
    #print(cv.get_feature_names())
    print(len(cv.get_feature_names()))


if __name__ == '__main__':
    main()
