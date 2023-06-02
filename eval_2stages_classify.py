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

# load mcode model
path = 'pths/' + df.iloc[0]['pth_name']
m_model = load_model(path, model_name, df.iloc[0]['size'], pretrained_model_config)

# load scode models
s_models = {}

for idx in range(1, len(df)) :
    if df.iloc[idx]['size'] > 1 :
        path = 'pths/' + df.iloc[idx]['pth_name']
        model = load_model(path, model_name, df.iloc[idx]['size'], 
                           pretrained_model_config)
        s_models[idx-1] = model
 

