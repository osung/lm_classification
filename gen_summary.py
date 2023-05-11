import os
import openai
import torch
import pandas as pd
from transformers import AutoTokenizer

openai.api_key = "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"

_my_linux_ = 1
THRESHOLD = 16
MAX_TOKEN = 128
TARGET_LENGTH = MAX_TOKEN * 2
MAX_PROMPT = 1024 #4096 - MAX_TOKEN

if _my_linux_  == 1:
   os.environ['CURL_CA_BUNDLE'] = '/home/osung/Downloads/kisti_cert.crt'

def get_encode_length(tokenizer, sentence) :
    encoded = tokenizer(sentence, padding=True, truncation=False)
    
    return len(encoded.input_ids)

prompt_one = "prompt: 다음 문장을 %d자 이내로 요약해줘: " % (TARGET_LENGTH)
print(prompt_one)

tokenizer = AutoTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
print(tokenizer)

train_df = pd.read_csv('/home/osung/data/korean/patent/train_mid2.tsv', sep='\t')
train_df = train_df.dropna()
train_df['summary'] = pd.Series(dtype='string')

print("train_df.shape:", train_df.shape)

#train_df = train_df[3:100]

#print(train_df)

for index, row in train_df.iterrows():
    encode_len = get_encode_length(tokenizer, row['text'])

    if encode_len > THRESHOLD and len(row['text']) > TARGET_LENGTH :
        prompt = prompt_one + row['text']
#print(prompt)
#print("len(prompt) :", len(prompt))
#print("len(word): ", len(prompt.split()))

        if len(prompt) > MAX_PROMPT :
            prompt = prompt[:MAX_PROMPT]
            print("truncated prompt:", len(prompt))

        response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=MAX_TOKEN, n=1, stop=None, temperature=1)

        train_df.at[index, 'summary'] = response.choices[0].text
        print(len(row['text']), len(response.choices[0].text))

    if index % 100 == 99 :
        train_df.to_csv("train_summary_" + str(index) + ".tsv", index=False, sep='\t')
#print(train_df)

print(train_df)

train_df.to_csv('train_summary.tsv', index=False, sep='\t')

