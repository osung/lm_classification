import os
import sys
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

#base_dir = '/home/osung/data/korean/commercial/csv/'
#filename = 'ad_summary.tsv'

base_dir = '/home/osung/data/korean/modu/json/'
filename = 'news_PEST_ksic_score_summary_no_dup'

df = pd.read_csv(base_dir+filename+'.tsv', sep='\t')
#df = df.dropna()

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
  
train_df.to_csv(base_dir+filename+'_train.tsv', sep='\t', index=False)
test_df.to_csv(base_dir+filename+'_test.tsv', sep='\t', index=False)

