import os
import pandas as pd
import json
import re

raw_dir = '/home01/hpc56a01/scratch/data/aihub/patent/validation/raw_data/'
label_dir = '/home01/hpc56a01/scratch/data/aihub/patent/validation/label_data/'

ksic_file = '/home01/hpc56a01/scratch/data/aihub/patent/ksic_code.csv'

ksic_df = pd.read_csv(ksic_file)
ksic_df['code'] = ksic_df['code'].apply(lambda x: f"{x:05}")

train_df = pd.DataFrame(columns=['text', 'KSIC', 'code'])

mid_codes = {}

for index, row in ksic_df.iterrows():
    print('Reading ', raw_dir+row['code']+'.json')

    mcode = row['code'][:-3] # truncate last 3 digits
    if not mcode in mid_codes.keys() :
        mid_codes[mcode] = len(mid_codes)

    print(mcode)

    with open(raw_dir+row['code']+'.json', 'r') as f:
        data = f.read()

    raw_json_data = json.loads(data)

    print('Reading ', label_dir+row['code']+'.json')

    with open(label_dir+row['code']+'.json', 'r') as f:
        data = f.read()

    label_json_data = json.loads(data)

    for label_data in label_json_data['dataset'] :
        f_list = list(filter(lambda x: x['documentId'] == label_data['documentId'], raw_json_data['dataset']))
        if 'claims' in f_list[0].keys() :
            #clean_text = re.sub('\([\u4e00-\u9fff0-9a-zA-Z]+\)', '', f_list[0]['claims'])
            #clean_text = re.sub('\(.*?\)', '', f_list[0]['claims'])
            clean_text = re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣\s]+', '', f_list[0]['claims'])
            new_row = pd.DataFrame({'text': clean_text, #f_list[0]['claims'], 
                                    'KSIC': row['code'], 
                                    #'code': index,
                                    'mcode': mcode,
                                    'code': mid_codes[mcode]}, index=[0])
            train_df = pd.concat([train_df, new_row], ignore_index=True)

train_df.to_csv('test_mid2.tsv', sep='\t', index=False)

print(mid_codes)

