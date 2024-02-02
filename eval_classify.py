import os
import torch
import argparse
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


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
    parser = argparse.ArgumentParser(description="Train classifiers using LLM.")

    parser.add_argument('pth_name', type=str, help='Set pth file to read')

    parser.add_argument('-te', '--test', type=str, help='Set test data')
    parser.add_argument('-d', '--dir', type=str, help='Set a base directory for the train and test data')
    parser.add_argument('-m', '--model', type=str, help='Set the base model for training')
    parser.add_argument('-v', '--variable', type=str, default='code', help='Set the target variable to infer')
    parser.add_argument('-b', '--batch', type=int, default=32, help='Set number of batchs for the training')
    parser.add_argument('-c', '--crt', type=str, help='Set the crt file for the certification')
    parser.add_argument('-n', '--num_labels', type=int, default=2, help='Set number of labels to classify')
    parser.add_argument('-C', '--code_csv', type=str, help='Set csv file describes target variable')
    parser.add_argument('-l', '--max_length', type=int, default=128, help='Set max length of the sentences')
    parser.add_argument('-t', '--truncate', type=int, default=10, help='Truncate sentences less than minimum length')
    parser.add_argument('--add_pad_token', action='store_true', help='Add PAD token to the tokenizer')
    parser.add_argument('-w', '--write', type=str, help='Set the file name to write incorrectly predicted data')

    args = parser.parse_args()

    return args

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('available device: ', device)
else:
    device = torch.device("cpu")
    print('available device: ', device)

# process commandline arguments
print("Processing commandline arguments");
args = get_args()

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

# set model
print("Setting model")

tokenizer = AutoTokenizer.from_pretrained(
    args.model, do_lower_case=False,
)

if args.add_pad_token :
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

target = args.variable

# load taget code 
if args.code_csv : 
    code_df = pd.read_csv(args.code_csv)
#print(code_df)

    num_labels = len(code_df) 

    codes = dict(zip(code_df[target], code_df['code']))

    print(codes)
else :
    num_labels = args.num_labels

print("num_labels is ", num_labels)

# load test data
test_df = pd.read_csv(test_path, sep='\t') #.head(50)
test_df = test_df.dropna()
test_df = test_df.reset_index(drop=True)

if not target in test_df.keys() :
    print('No', target, 'in the loaded dataframe')
    quit()

if target != 'code' :
    targets = []  # list

    for idx, row in test_df.iterrows() :
        targets.append(codes[row[target]])

    test_df['code'] = targets

# 'text' column의 문자열 길이가 args.truncate 이하인 row 삭제
if args.truncate > 1 :
    print("truncate sentences less than minimum length", args.truncate)
    test_df = test_df[test_df['text'].str.len() >= args.truncate]
    test_df = test_df.reset_index(drop=True)

#test_df = test_df.drop(columns=['Unnamed: 0'])
print(test_df)

pretrained_model_config = AutoConfig.from_pretrained(args.model)
pretrained_model_config.num_labels = num_labels #44 #(mid) #118 (small)  #564 
model = AutoModelForSequenceClassification.from_pretrained(
    args.model,
    config=pretrained_model_config,
)

if args.add_pad_token :
    model.config.pad_token_id = model.config.eos_token_id

# parallelization
if torch.cuda.device_count() > 1:
    print(f'Using {torch.cuda.device_count()} GPUs.')

    model = torch.nn.DataParallel(model)  

model = model.to(device)

# load fine-tuned model
loaded_state_dict = torch.load(args.pth_name)
model.load_state_dict(loaded_state_dict, strict=False) #, map_location='cuda:0')

#new_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
#model.load_state_dict(new_state_dict)

# evaluation
print("Preparing test data")

print("Tokenizing test data")

test_input_ids, test_attention_masks, test_labels = get_encode_data(tokenizer, test_df['text'].tolist(), test_df['code'], max_length=args.max_length)

print("Generating torch tensor from the tokenized test data")

test_dataset = TrainDataset(test_input_ids, test_attention_masks, test_labels)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

print("Evaluating model using test data")

model.eval()

y_true = []
y_pred = []

for batch in tqdm(test_dataloader, desc='Evaluating', leave=False):
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)

    with torch.no_grad():
        outputs = model(b_input_ids, attention_mask=b_input_mask)

#print(outputs)

    logits = outputs.logits
    _, predicted = torch.max(logits, dim=1)

    y_true.extend(b_labels.tolist())
    y_pred.extend(predicted.tolist())
    

y_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})

correct = len(y_df[y_df['y_true'] == y_df['y_pred']])
print(f'Correct num: {correct}')

incorrect = len(y_df[y_df['y_true'] != y_df['y_pred']])
print(f'Incorrect num: {incorrect}')


cal_accuracy = correct / len(y_df)

print(f'Calculated Accuracy: {cal_accuracy}')

accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy}')

if args.write is not None :
    test_df['prediction'] = y_df['y_pred']
#test_df['true'] = y_df['y_true']
    incorrect_df = test_df[test_df['prediction'] != test_df['code']]
    incorrect_df.to_csv(base_dir + '/' + args.write, sep='\t')


