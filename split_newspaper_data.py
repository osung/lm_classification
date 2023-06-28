import os
import sys
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

pest_code = {
    '정치': 0,
    '경제': 1,
    '사회': 2,
    'IT/과학': 3,
    '기타': -1}

filename = sys.argv[1]

df = pd.read_csv(filename, sep='\t')
df = df.dropna()

train_list = []
test_list = []

for code in tqdm(range(0,4)) :
    pest_df = df[df['code'] == code]

    train_df, test_df = train_test_split(pest_df, test_size=0.2, random_state=42)
    train_list.append(train_df)
    test_list.append(test_df)

train_df = pd.concat([train_list[0], train_list[1], train_list[2], train_list[3]])
test_df = pd.concat([test_list[0], test_list[1], test_list[2], test_list[3]])
   
train_df.to_csv('train.tsv', sep='\t', index=False)
test_df.to_csv('test.tsv', sep='\t', index=False)

