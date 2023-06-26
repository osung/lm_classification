import os
import csv
import torch
import argparse
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import ElectraTokenizer, ElectraForSequenceClassification, ElectraConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from accelerate.utils import MegatronLMDummyScheduler


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __getitem__(self, idx):
        input_id = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        label = self.labels[idx]

        return input_id, attention_mask, label

    def __len__(self):
        return len(self.input_ids)


def get_encode_data(tokenizer, sentences, labels, max_length=128):
    encoded_inputs = tokenizer(sentences, padding=True, truncation=True, max_length=max_length)
    input_ids = torch.tensor(encoded_inputs['input_ids'])
    attention_masks = torch.tensor(encoded_inputs['attention_mask'])
    labels = torch.tensor(labels)
    
    return input_ids, attention_masks, labels


def get_args() :
    parser = argparse.ArgumentParser(description="Test trained classifiers based on LLM.")
    parser.add_argument('model_dir', type=str, help='Set full path of directory contains saved model')

    parser.add_argument('-te', '--test', type=str, help='Set test data')
    parser.add_argument('-d', '--dir', type=str, help='Set a base directory for the train and test data')
    parser.add_argument('-m', '--model', type=str, help='Set the base model for training')
    parser.add_argument('-v', '--variable', type=str, default='code', help='Set the target variable to learn')
    parser.add_argument('-b', '--batch', type=int, default=32, help='Set number of batchs for the training')
    parser.add_argument('-c', '--crt', type=str, help='Set the crt file for the certification')
    parser.add_argument('-n', '--num_labels', type=int, default=2, help='Set number of labels to classify')
    parser.add_argument('-l', '--max_length', type=int, default=128, help='Set max length of the sentences')
    parser.add_argument('-t', '--truncate', type=int, default=10, help='Truncate sentences less than minimum length')
    parser.add_argument('--add_pad_token', action='store_true', help='Add PAD token to the tokenizer')

    args = parser.parse_args()

    return args


def main():

    accelerator = Accelerator()
    device = accelerator.device

    print(accelerator)
    print(device)

    # process commandline arguments
    print("Processing commandline arguments");
    args = get_args()

    print(args)

    if args.dir is None :
        base_dir = '.'
    else :
        base_dir = args.dir

    if args.test is None :
        test_path = None
    else :
        test_path = base_dir + '/' + args.test

    model_name = args.model.replace('/', '_')

    if args.crt is not None :
        os.environ['CURL_CA_BUNDLE'] = args.crt
        
    # evaluation
    print("Preparing test data")
    
    test_df = pd.read_csv(test_path, sep='\t')
    test_df = test_df.dropna()
    test_df = test_df.reset_index(drop=True)
    
    # 'text' column의 문자열 길이가 args.truncate 이하인 row 삭제
    if args.truncate > 1 :
        test_df = test_df[test_df['text'].str.len() >= args.truncate]
        test_df = test_df.reset_index(drop=True)
    
    print(test_df)

    target = args.variable
    
    if not target in test_df.keys() :
        print('No', target, 'in the loaded dataframe')
        quit()
    
    target_idx = None
        
    if target == 'mcode' :
        target_idx = 'midx'
    elif target == 'scode' :
        target_idx = 'sidx'
 
    # set model
    print("Setting model")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, do_lower_case=False,
    )
    
    if args.add_pad_token :
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    
    pretrained_model_config = AutoConfig.from_pretrained(args.model)
    
    pretrained_model_config.num_labels = args.num_labels #test_df[target].nunique()
    print("num_labels :", pretrained_model_config.num_labels)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        config=pretrained_model_config,
    )
    
    if args.add_pad_token :
        model.config.pad_token_id = model.config.eos_token_id

    # tokenize test data
    print("Tokenizing test data")
    
    test_input_ids, test_attention_masks, test_labels = get_encode_data(tokenizer, test_df['text'].tolist(), test_df[target_idx], max_length=args.max_length)
    
    print("Generating torch tensor from the tokenized test data")
    
    test_dataset = TrainDataset(test_input_ids, test_attention_masks, test_labels)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch)

    # set the model up
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    #lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    #model, _, test_dataloader, _ = accelerator.prepare(model, optimizer, test_dataloader, lr_scheduler=None)
    model, _, test_dataloader = accelerator.prepare(model, optimizer, test_dataloader) #, lr_scheduler=None)

    #model_dir = os.path.basename(args.model_dir)
    model_dir = args.model_dir
    print("model dir is", model_dir)
        
    # load pretrained model
    #loaded_state_dict = torch.load(resume_name)
    accelerator.load_state(model_dir)
    
    # evaluating trained model
    model.eval()
    
    y_true = []
    y_pred = []
    
    for batch in tqdm(test_dataloader, desc='Evaluating', leave=False):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
    
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)
    
        logits = outputs.logits
        _, predicted = torch.max(logits, dim=1)
    
        y_true.extend(b_labels.tolist())
        y_pred.extend(predicted.tolist())
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy}')
   
if __name__ == "__main__":
    main()

