import os
import sys
import pandas as pd
import json
from tqdm import tqdm

pest_code = {
    '정치': 0,
    '경제': 1,
    '사회': 2,
    'IT/과학': 3,
    '기타': -1}

THRESHOLD = 30


def get_clean_dicts(df) :
    news_list = []

    for _, row in df.iterrows() :
        if row['code'] >= 0 and len(row['text']) > THRESHOLD :
            a_dict = {}
            a_dict['text'] = row['text']
            a_dict['category'] = row['category']
            a_dict['code'] = row['code']

            news_list.append(a_dict)

    return news_list


filenames = sys.argv[1:]
print("filenames are", filenames)

total_list = []

for filename in tqdm(filenames) :
    df = pd.read_csv(filename, sep='\t')
    df = df.dropna()
    #print(df)

    total_list = total_list + get_clean_dicts(df)

news_df = pd.DataFrame(total_list)
news_df.to_csv('news_merged.tsv', sep='\t', index=False)

         

