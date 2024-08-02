import openai
import os

openai.api_key = "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"

_my_linux_ = 1

if _my_linux_  == 1:
   os.environ['CURL_CA_BUNDLE'] = '/home/osung/Downloads/kisti_cert.crt'

system_message = "You are a friendly assistant. Respond using Korean language."
prompt = "Hello, how are you doing today?"

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": prompt}
]

response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=messages,
            temperature=0.0,  # 창의성을 조절하는 옵션
            #max_tokens=max_tokens,  # 답변의 최대 토큰 수 설정
           )

answer = response['choices'][0]['message']['content']

print(answer)

