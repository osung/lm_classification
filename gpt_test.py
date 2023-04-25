import torch
from torch.optim import AdamW
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

model_name = 'gpt2' #'/home01/hpc56a01/scratch/kogpt2-base-v2'

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.init()
    print('available device: ', device)
else:
    device = torch.device("cpu")
    print('available device: ', device)

# Load pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

model = GPT2ForSequenceClassification.from_pretrained(model_name)
model.config.pad_token_id = model.config.eos_token_id

model = model.to(device)

# Define the optimizer and learning rate
optimizer = AdamW(model.parameters(), lr=1e-5)

# Prepare the training data
train_data = ["Example sentence 1", "Example sentence 2"]
train_labels = [1, 0]

# Tokenize the training data
train_encodings = tokenizer(train_data, truncation=True, padding=True)

# Convert the labels to PyTorch tensors
#train_labels = torch.tensor(train_labels)

# Create a PyTorch DataLoader for training
train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_encodings['input_ids']).to(device), 
        torch.tensor(train_encodings['attention_mask']).to(device), 
        torch.tensor(train_labels).to(device))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

# Fine-tune the model
model.train()

for epoch in range(3):  # 3 epochs
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1} loss: {loss.item()}")

# Save the fine-tuned model
model.save_pretrained("gpt2_finetuned")

