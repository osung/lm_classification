import os
import sys
import pandas as pd
from tqdm import tqdm
import re

filename = sys.argv[1]

df = pd.read_csv(filename, sep='\t')
clean_text_list = []

for row in tqdm(df.itertuples(index=False), total=len(df)) :
    clean_text = re.sub(r'[^A-Za-z가-힣0-9\s]', '', row.text)
    clean_text_list.append(clean_text)

df['text'] = clean_text_list

name = filename.rsplit(".", 1)[0]

df.to_csv(name + '_clean.tsv', sep='\t', index=False)

