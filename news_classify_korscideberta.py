import os
import csv
import torch
import argparse
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from tokenization_korscideberta_v2 import DebertaV2Tokenizer
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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

    parser.add_argument('-tr', '--train', type=str, required=True, help='Set train data (mandatory)')
    parser.add_argument('-te', '--test', type=str, help='Set test data')
    parser.add_argument('-d', '--dir', type=str, help='Set a base directory for the train and test data')
    parser.add_argument('-v', '--variable', type=str, default='code', help='Set the target variable to learn')
    parser.add_argument('-e', '--epoch', type=int, default=5, help='Set number of epochs for the training')
    parser.add_argument('-b', '--batch', type=int, default=32, help='Set number of batchs for the training')
    parser.add_argument('-c', '--crt', type=str, help='Set the crt file for the certification')
    parser.add_argument('-n', '--num_labels', type=int, default=2, help='Set number of labels to classify')
    parser.add_argument('-l', '--max_length', type=int, default=128, help='Set max length of the sentences')
    parser.add_argument('-t', '--truncate', type=int, default=10, help='Truncate sentences less than minimum length')
    parser.add_argument('-r', '--resume', type=str, help='Set pth file to resume')
    parser.add_argument('--add_pad_token', action='store_true', help='Add PAD token to the tokenizer')
    parser.add_argument('--save_every_iter', action='store_true', help='Set if want to store pth files for every iteration')

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

print(args)

if args.dir is None :
    base_dir = '.'
else :
    base_dir = args.dir

train_path = base_dir + '/' + args.train

if args.test is None :
    test_path = None
else :
    test_path = base_dir + '/' + args.test

args_model = 'kisti/korscideberta'
model_name = args_model.replace('/', '_')


if args.crt is not None :
    os.environ['CURL_CA_BUNDLE'] = args.crt


# load data
print("Preparing train data")

train_df = pd.read_csv(train_path, sep='\t')
train_df = train_df.dropna()
train_df = train_df.reset_index(drop=True)

#target = 'code'
target = args.variable

if not target in train_df.keys() :
    print('No', target, 'in the loaded dataframe')
    quit()

if target != 'code' :
    codes = {}    # dict to store pairs of KSIC and int code to learn
    targets = []  # list

    for idx, row in train_df.iterrows() :
        if not row[target] in codes.keys() :
            codes[row[target]] = len(codes)

        targets.append(codes[row[target]])

    train_df['code'] = targets

    # write code to the csv file
    filename =  model_name + '_' + args.train + '_' + target +'_code.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for key, value in codes.items():
            writer.writerow([key, value])

# set model
print("Setting model")

#tokenizer = AutoTokenizer.from_pretrained(
#   args_model, do_lower_case=False,
#)


tokenizer = DebertaV2Tokenizer.from_pretrained(args_model)

if args.add_pad_token :
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

pretrained_model_config = AutoConfig.from_pretrained(args_model)

pretrained_model_config.num_labels = train_df['code'].nunique()
print("num_labels :", pretrained_model_config.num_labels)

model = AutoModelForSequenceClassification.from_pretrained(args_model, config=pretrained_model_config)

#model = AutoModelForSequenceClassification.from_pretrained(
#    args_model,
#    config=pretrained_model_config,
#)

if args.add_pad_token :
    model.config.pad_token_id = model.config.eos_token_id


# parallelization
if torch.cuda.device_count() > 1:
    #print(f'Using {torch.cuda.device_count()} GPUs.')

    model = torch.nn.DataParallel(model)  

model = model.to(device)

# process resume file if any
resume_no = 0

if args.resume is None :
    pth_name = model_name + '_' + args.train + '_' + target + '_b' + str(args.batch) + '_e' + str(args.epoch) + '_ml' + str(args.max_length) + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.pth'
else :
    resume_name = os.path.basename(args.resume)
    pth_name = os.path.splitext(resume_name)[0] + '.pth'

    print("resume_name is", resume_name)
    print("pth_name is", pth_name)
    
    last_underscore_index = resume_name.rfind("_")
    if last_underscore_index != -1:
        resume_no = int(resume_name[last_underscore_index +1:]) + 1

    print("resume no is", resume_no)

    # load pretrained model
    loaded_state_dict = torch.load(resume_name)
    model.load_state_dict(loaded_state_dict)

# 'text' column의 문자열 길이가 args.truncate 이하인 row 삭제
if args.truncate > 1 :
    train_df = train_df[train_df['text'].str.len() >= args.truncate]    
    train_df = train_df.reset_index(drop=True)

print(train_df)

print("Tokenizing train data")

train_input_ids, train_attention_masks, train_labels = get_encode_data(tokenizer, train_df['text'].tolist(), train_df['code'], max_length=args.max_length)

print("Generating torch tensor from the tokenized train data")

train_dataset = TrainDataset(train_input_ids, train_attention_masks, train_labels)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True)

# fine tuning
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = args.epoch
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

model.train()

print("range is", resume_no, ", ", num_epochs)
print(range(resume_no, num_epochs))

for epoch in range(resume_no, num_epochs):
    epoch_loss = 0
    epoch_steps = 0

    for step, batch in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch+1}', leave=False)):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()

        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

        loss = outputs.loss.mean()
        logits = outputs.logits

        loss.backward()

        optimizer.step()
        lr_scheduler.step()

        epoch_loss += loss.item()
        epoch_steps += 1

        if step % 1000 == 0:
            print(f'Epoch {epoch+1} / Step {step+1} - Loss: {epoch_loss/epoch_steps:.5f}')

    if args.save_every_iter :
        formatted_epoch = '_%02d' % epoch
        print('saved pth file :', pth_name+formatted_epoch)
        torch.save(model.state_dict(), pth_name+formatted_epoch) # save pth at every epoch

torch.save(model.state_dict(), pth_name)


