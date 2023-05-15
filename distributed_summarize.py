import os
import sys
import torch
import torch.distributed as dist
import pandas as pd
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pickle

MAX_TOKEN = 256
THRESHOLD = 256

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
                           num_beams=4, )

    response = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)

    #print(len(input_text), len(response))

    return response

cont_idx = [0, 0, 0, 0] #10000, 8000, 8000, 9000]
cont_file = [] #'summary_data_0_9999.pkl', 'summary_data_1_7999.pkl', 'summary_data_2_7999.pkl', 'summary_data_3_8999.pkl']

os.environ['CURL_CA_BUNDLE'] = '/home/osung/Downloads/kisti_cert.crt'
model_name = 'eenzeenee/t5-base-korean-summarization'
#model_name = 'psyche/KoT5-summarization'


# 분산 학습을 위해 초기화
dist.init_process_group(backend='nccl')

world_size = dist.get_world_size()
rank = dist.get_rank()
print("Rank:", rank, "out of", world_size)

num_gpus = torch.cuda.device_count()
print("Number of GPUs:", num_gpus)

# 각 프로세스에 할당될 CUDA 장치 인덱스 계산
device_indices = list(range(rank, num_gpus, world_size))

# 각 프로세스에 CUDA 장치 할당
device = torch.device(f"cuda:{device_indices[0]}")
torch.cuda.set_device(device)

'''
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('available device: ', device)
else:
    device = torch.device("cpu")
    print('available device: ', device) '''


# 모델 및 토크나이저 초기화
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = model.to(device)

print("loading model", model_name, "is done")

# DistributedDataParallel로 모델 감싸기
model = DistributedDataParallel(model)

# 데이터 로딩
print("Loading tsv data")
#df = pd.read_csv('/home01/hpc56a01/scratch/data/aihub/patent/train_mid2.tsv', sep='\t')
#df = pd.read_csv('/home/osung/data/korean/patent/train_mid2.tsv', sep='\t')
df = pd.read_csv('/home/osung/data/korean/patent/test_mid2.tsv', sep='\t')
df = df.dropna()
df = df.reset_index(drop=True)
print("Done")

#df = df[:1000]

# 데이터 분할
data_per_gpu = len(df) // world_size

df = df[data_per_gpu * rank: data_per_gpu * (rank + 1)]

print("[", rank, "] data_per_gpu:", data_per_gpu, ", len(df):", len(df))

dist.barrier()

#print(df)

# 분할된 데이터를 각 프로세스로 전달
#dist.broadcast(df, src=0)

# set output file name without extension
out_name = 'summary_test_' + str(rank) 
#print("Length of df in rank", rank, "is", len(df))

# 각 프로세스에서 동일한 작업 수행
outputs = []
start_idx = 0

if len(cont_file) >= world_size :
    with open(cont_file[rank], 'rb') as file:
        outputs = pickle.load(file)

    start_idx = cont_idx[rank]
    print("[", rank, "] load texts:", len(outputs), "start idx:", start_idx)

#for text in df['text']:
for idx, row in df[start_idx:].iterrows() :
    index = idx - rank * data_per_gpu
    text = row['text']

    # 입력 텍스트를 토크나이징
    try :
        input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    except TypeError:
        print("ERROR RANK [", rank, "] INDEX ", index, "TEXT", text)
        sys.exit()

    if len(input_ids[0]) > MAX_TOKEN :
        # generate() 함수를 사용하여 텍스트 생성
        with torch.no_grad():
            summary = summarize(text, tokenizer, model.module)
            #print("[", rank, "] index: ", index, "(", len(text), len(summary), ")")
            if len(summary) > len(text) :
                summary = text
    else :
        summary = text

    # 결과 저장
    outputs.append(summary)

    if index % 10 == 0 :
        print(rank, ":", index, "out of", len(df))

    if index % 1000 == 999 :
        file_name = out_name + '_' + str(index) + '.pkl'   

        with open(file_name, "wb") as file:
            pickle.dump(outputs, file)

print("[", rank, "]", "Summarization is done")

df['summary'] = outputs
csv_name = out_name + '.tsv'
df.to_csv(csv_name, index=False, sep='\t')

# 결과 수집
#all_outputs = dist.gather(torch.tensor(outputs), dst=0)

# 결과를 프로세스 0에서 출력
#if dist.get_rank() == 0:

#    df['summary'] = all_outputs

#    df.to_csv("summary_data.tsv", index=False, sep='\t')
    
    #for output in all_outputs:
        #print(output)

