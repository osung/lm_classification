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

raw_dir = '/home/osung/data/korean/modu/modu_newspaper/'

news_dicts = []

filename = raw_dir + sys.argv[1]
print('filename:', filename)

f = open(filename, 'r')
data = f.read()

json_data = json.loads(data)
doc_list = json_data['document']

#print("Number of doc is", len(doc_list))

for doc in tqdm(doc_list) :
    category = doc['metadata']['topic']

    if category in pest_code.keys() :
        code = pest_code[category]
    else :
        code = pest_code['기타']

    articles = doc['paragraph']
#print(step, ": category is", category, '# of articles is', len(articles))

    for a in articles :
        a_dict = {}
        a_dict['text'] = a['form']
        a_dict['category'] = category
        a_dict['code'] = code

        news_dicts.append(a_dict)

news_df = pd.DataFrame(news_dicts)

name, _ = os.path.splitext(filename)
outname = name + '.tsv'
#print('Outfile:', outname)

news_df.to_csv(outname, sep='\t', index=False)


