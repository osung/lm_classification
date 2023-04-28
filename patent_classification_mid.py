import os
import torch
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

_my_linux_=0   #if not neuron, set it 0

if _my_linux_ == 1 :
    os.environ['CURL_CA_BUNDLE'] = '/home/osung/Downloads/kisti_cert.crt'
else :
    train_path='/home01/hpc56a01/scratch/data/aihub/patent/train_mid.tsv'
    test_path='/home01/hpc56a01/scratch/data/aihub/patent/test_mid.tsv'

# default batch size
BATCH_SIZE = 128

model_name='monologg/koelectra-base-v3-discriminator'
#model_name='beomi/KcELECTRA-base'
#model_name='skt/kobert-base-v1'
#model_name='beomi/kcbert-base'
#model_name='beomi/kobert'
#model_name='krevas/finance-koelectra-base-discriminator'    

if model_name == 'monologg/koelectra-base-v3-discriminator':
    pth_name='koelectra3_patent_20.pth'

    if _my_linux_ == 1:
        BATCH_SIZE = 128
    else :
        BATCH_SIZE = 2048

elif model_name=='krevas/finance-koelectra-base-discriminator':
    pth_name='finkoelectra_patent.pth'

    if _my_linux_ == 1:
        BATCH_SIZE = 128
    else :
        BATCH_SIZE = 512

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

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('available device: ', device)
else:
    device = torch.device("cpu")
    print('available device: ', device)

tokenizer = AutoTokenizer.from_pretrained(
    model_name, do_lower_case=False,
)

pretrained_model_config = AutoConfig.from_pretrained(model_name)
pretrained_model_config.num_labels = 44 #(mid) #118 (small)  #564 

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    config=pretrained_model_config,
)

print("Preparing train data")

train_df = pd.read_csv(train_path, sep='\t')
print(train_df)

print("Tokenizing train data")

train_input_ids, train_attention_masks, train_labels = get_encode_data(tokenizer, train_df['text'].tolist(), train_df['code'])

print("Generating torch tensor from the tokenized train data")

train_dataset = TrainDataset(train_input_ids, train_attention_masks, train_labels)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# parallelization
if torch.cuda.device_count() > 1:
    print(f'Using {torch.cuda.device_count()} GPUs.')

    model = torch.nn.DataParallel(model)  

model = model.to(device)


# load states
loaded_state_dict = torch.load('koelectra3_patent.pth')
#print(loaded_state_dict.keys())

model.load_state_dict(loaded_state_dict)


# fine tuning
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 10
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
print(test_df)

print("Tokenizing train and val data")

test_input_ids, test_attention_masks, test_labels = get_encode_data(tokenizer, test_df['text'].tolist(), test_df['code'], max_length=256)

print("Generating torch tensor from the tokenized test data")

test_dataset = TrainDataset(test_input_ids, test_attention_masks, test_labels)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

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

