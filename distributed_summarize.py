import torch
import torch.distributed as dist
import pandas as pd
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MAX_TOKEN = 256

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

#os.environ['CURL_CA_BUNDLE'] = '/home/osung/Downloads/kisti_cert.crt'
model_name = 'eenzeenee/t5-base-korean-summarization'

# 분산 학습을 위해 초기화
torch.distributed.init_process_group(backend='nccl')

# 모델 및 토크나이저 초기화
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# DistributedDataParallel로 모델 감싸기
model = DistributedDataParallel(model)

# 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 데이터 로딩
df = pd.read_csv('/home01/hpc56a01/scratch/data/aihub/patent/train_mid2.tsv', sep='\t')

df = df[:100]

# 데이터 분할
data_per_gpu = len(df) // torch.distributed.get_world_size()
df = df[data_per_gpu * torch.distributed.get_rank(): data_per_gpu * (torch.distributed.get_rank() + 1)]

# 분할된 데이터를 각 프로세스로 전달
dist.broadcast(df, src=0)

# 각 프로세스에서 동일한 작업 수행
outputs = []
for text in df['text']:
    # 입력 텍스트를 토크나이징
    #input_ids = tokenizer.encode(text, return_tensors='pt').to(device)

    # generate() 함수를 사용하여 텍스트 생성
    with torch.no_grad():
        output = summarize(text, tokenizer, model)
        #model.generate(input_ids, max_length=100)

        # 생성된 텍스트를 디코딩
        #generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # 결과 저장
        outputs.append(output)

# 결과 수집
all_outputs = torch.distributed.gather(torch.tensor(outputs), dst=0)

# 결과를 프로세스 0에서 출력
if torch.distributed.get_rank() == 0:

    df['summary'] = all_outputs

    df.to_csv("summary_data.tsv", index=False, sep='\t')
    
    #for output in all_outputs:
        #print(output)

