import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd

MAX_TOKEN = 256

def get_encode_length(tokenizer, sentence) :
    encoded = tokenizer(sentence, padding=True, truncation=False)

    return len(encoded.input_ids)


def get_encode_data(tokenizer, sentences):
    encoded_inputs = tokenizer(sentences, padding=True, truncation=False)
    input_ids = torch.tensor(encoded_inputs['input_ids'])
    attention_masks = torch.tensor(encoded_inputs['attention_mask'])

    return input_ids, attention_masks


def summarize(input_text, tokenizer, model) :
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate Summary Text Ids
    summary_text_ids = model.generate(
        input_ids=input_ids,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        length_penalty=2.0,
        max_new_tokens=200,
        min_new_tokens=100,
        no_repeat_ngram_size=2,
        num_beams=4,
    )
    
    response = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)
    
    #print(len(input_text), len(response))
    
    return response

#os.environ['CURL_CA_BUNDLE'] = '/home/osung/Downloads/kisti_cert.crt'
model_name = 'eenzeenee/t5-base-korean-summarization'

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('available device: ', device)
else:
    device = torch.device("cpu")
    print('available device: ', device) 

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# parallelization
if torch.cuda.device_count() > 1:
    print(f'Using {torch.cuda.device_count()} GPUs.')

    model = torch.nn.DataParallel(model) 

model = model.to(device)

train_df = pd.read_csv('/home01/hpc56a01/scratch/data/aihub/patent/train_mid2.tsv', sep='\t')
train_df = train_df.dropna()
train_df = train_df.reset_index(drop=True)

train_df = train_df[train_df['text'].str.len() >= 300]
train_df = train_df.reset_index(drop=True)

#train_df['summary'] = pd.Series(dtype='string')

print(train_df)

train_input_ids, train_attention_masks = get_encode_data(tokenizer, train_df['text'].tolist())

print("Generating toch tensor from the tokenized data")



count = 0
for index, row in train_df.iterrows():
    encode_len = get_encode_length(tokenizer, row['text'])

    if encode_len > MAX_TOKEN :
        response = summarize(row['text'], tokenizer, model.module)
        train_df.at[index, 'summary'] = response
        count += 1

    if index % 10 == 9 :
        print(str(index), 'of', str(len(train_df)), ':', str(count), 'sentences are summarized so far')

    if index % 100 == 99 :
        train_df.to_csv("train_summary_" + str(index) + ".tsv", index=False, sep='\t')

print(train_df)

train_df.to_csv('train_summary.tsv', index=False, sep='\t')


