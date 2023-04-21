from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from ratsnlp.nlpbook.classification import NsmcCorpus, ClassificationDataset
from ratsnlp.nlpbook.classification import ClassificationTrainArguments
from tokenizers import BertWordPieceTokenizer
from Korpora import Korpora
import torch
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

BATCH_SIZE = 512

class NSMCDataset(torch.utils.data.Dataset):
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


os.environ['CURL_CA_BUNDLE'] = '/home/osung/Downloads/kisti_cert.crt'
#model_name='/home/osung/models/huggingface/kcbert-base'  #'beomi/kcbert_base'
#model_name='beomi/KcELECTRA-base'
#model_name='skt/kobert-base-v1'
model_name='beomi/kobert'

pth_name='kobert_nsmc.pth'

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('현재 사용 가능한 device: ', device)
else:
    device = torch.device("cpu")
    print('현재 사용 가능한 device: ', device)

tokenizer = BertTokenizer.from_pretrained(
    model_name, do_lower_case=False,
)

pretrained_model_config = BertConfig.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name,
    config=pretrained_model_config,
)

# 병렬처리
if torch.cuda.device_count() > 1:
    print(f'Using {torch.cuda.device_count()} GPUs.')

    model = torch.nn.DataParallel(model)  

model = model.to(device)

nsmc = Korpora.load("nsmc")

print("Preparing train and val data")

train_df = pd.DataFrame({"text": nsmc.train.texts, 'label': nsmc.train.labels})
#train_inputs, val_inputs, train_labels, val_labels = train_test_split(train_df['text'], train_df['label'], random_state=59, test_size=0.1)

print(train_df)

print("Tokenizing train and val data")

train_input_ids, train_attention_masks, train_labels = get_encode_data(tokenizer, train_df['text'].tolist(), train_df['label'])

#train_input_ids, train_attention_masks, train_labels = get_encode_data(tokenizer, train_inputs.tolist(), train_labels)
#val_input_ids, val_attention_masks, val_labels = get_encode_data(tokenizer, val_inputs.tolist(), val_labels)

print("Generating torch tensor from the tokenized train and val data")

train_dataset = NSMCDataset(train_input_ids, train_attention_masks, train_labels)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

#val_dataset = NSMCDataset(val_input_ids, val_attention_masks, val_labels)
#val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32)


# fine tuning

optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 5
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


torch.save(model.state_dict(), pth_name) #'bert_nsmc.pth')


# evaluation

print("Preparing test data")

test_df = pd.DataFrame({"text": nsmc.test.texts, 'label': nsmc.test.labels})

print("Tokenizing test data")

input_ids, attention_masks, labels = get_encode_data(tokenizer, test_df['text'].tolist(), test_df['label'])

print("Generating torch tensor from the tokenized data")

test_dataset = NSMCDataset(input_ids, attention_masks, labels)
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



