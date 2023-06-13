import os
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


def load_model(path, model_name, num_labels, config) :

    config.num_labels = num_labels
    model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, 
                    config=config, )
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    loaded_state_dict = torch.load(path)
    model.load_state_dict(loaded_state_dict)

    return model


def replace_model(model, path, num_labels) :

    model.module.config.num_labels = num_labels

    in_features = model.module.classifier.out_proj.in_features
    model.module.classifier.out_proj = torch.nn.Linear(in_features, 
                                              model.module.config.num_labels)

    loaded_state_dict = torch.load(path)
    model.load_state_dict(loaded_state_dict)

    return model



os.environ['CURL_CA_BUNDLE'] = '/home/osung/Downloads/kisti_cert.crt'
model_name = 'klue/roberta-large'

device = torch.device('cuda')

tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower=False,)
pretrained_model_config = AutoConfig.from_pretrained(model_name)


df = pd.read_csv('klue_roberta_pth_list.csv')

print(df)

# LOAD ALL MODELS SIMULTANEOUSLY
models = []

for idx, row in df.iterrows() :
    if row['size'] > 1 :
        path = 'pths/' + row['pth_name']
        model = load_model(path, model_name, row['size'], pretrained_model_config)
        models.append(model)
    else :
        models.append(None)

print(len(models))

df['model'] = models

'''
print(df.iloc[0]['model'])
print(df.iloc[3]['model'])
print(df.iloc[5]['model'])
print(df.iloc[8]['model']) '''


'''
# LOAD ONE MODEL ONLY AND REPLACE STATES IN REAL TIME
# load the first model (mcode model)
path = 'pths/' + df.iloc[0]['pth_name']
model = load_model(path, model_name, df.iloc[0]['size'], pretrained_model_config)
print(model)

# load following models (for scode)
for idx in range(1, len(df)) :
    if df.iloc[idx]['size'] > 1 :
        path = 'pths/' + df.iloc[idx]['pth_name']
        model = replace_model(model, path, df.iloc[idx]['size'])

print(model) '''

       

