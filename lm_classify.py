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
from transformers import ElectraTokenizer, ElectraForSequenceClassification, ElectraConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Korpora import Korpora


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

    parser.add_argument('-m', '--model', type=str, help='Set the base model for training')
    parser.add_argument('-e', '--epoch', type=int, default=5, help='Set number of epochs for the training')
    parser.add_argument('-b', '--batch', type=int, default=32, help='Set number of batchs for the training')
    parser.add_argument('-c', '--crt', type=str, help='Set the crt file for the certification')
    parser.add_argument('-n', '--num_labels', type=int, default=2, help='Set number of labels to classify')

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

train_path = base_dir + '/' + args.train

if args.test is None :
    test_path = None
else :
    test_path = base_dir + '/' + args.test

model_name = args.model.replace('/', '_')

pth_name = model_name + '_' + args.train + '_b' + str(args.batch) + '_e' + str(args.epoch) + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.pth'

# set model
print("Setting model")

tokenizer = AutoTokenizer.from_pretrained(
    args.model, do_lower_case=False,
)

pretrained_model_config = AutoConfig.from_pretrained(args.model)
pretrained_model_config.num_labels = args.num_labels #44 #(mid) #118 (small)  #564 
model = AutoModelForSequenceClassification.from_pretrained(
    args.model,
    config=pretrained_model_config,
)

print("Preparing train data")

train_df = pd.read_csv(train_path, sep='\t')
train_df = train_df.dropna()
train_df = train_df.reset_index(drop=True)
print(train_df)

print("Tokenizing train data")

train_input_ids, train_attention_masks, train_labels = get_encode_data(tokenizer, train_df['text'].tolist(), train_df['code'])

print("Generating torch tensor from the tokenized train data")

train_dataset = TrainDataset(train_input_ids, train_attention_masks, train_labels)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True)


# parallelization
if torch.cuda.device_count() > 1:
    print(f'Using {torch.cuda.device_count()} GPUs.')

    model = torch.nn.DataParallel(model)  

model = model.to(device)

# fine tuning
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = args.epoch
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

model.train()

for epoch in range(num_epochs):
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


torch.save(model.state_dict(), pth_name)
#model.save_pretrained("patent_koelastic")

# evaluation
print("Preparing test data")

test_df = pd.read_csv(test_path, sep='\t')
test_df = test_df.dropna()
test_df = test_df.reset_index(drop=True)
print(test_df)

print("Tokenizing test data")

test_input_ids, test_attention_masks, test_labels = get_encode_data(tokenizer, test_df['text'].tolist(), test_df['code'], max_length=256)

print("Generating torch tensor from the tokenized test data")

test_dataset = TrainDataset(test_input_ids, test_attention_masks, test_labels)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch)

print("Evaluating model using test data")

model.eval()

y_true = []
y_pred = []

for batch in tqdm(test_dataloader, desc='Evaluating', leave=False):
#for batch in test_dataloader:
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

