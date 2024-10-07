import openai
import os

openai.api_key = "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"

_my_linux_ = 1

if _my_linux_  == 1:
   os.environ['CURL_CA_BUNDLE'] = '/home/osung/Downloads/kisti_cert.crt'

system_message = "You are a friendly assistant. Respond using Korean language."

prompt = "다음 문장을 200자로 짧게 요약해줘. "

sentence = "동진섬유는 글로벌 스포츠 브랜드 기업에 신발 원단을 공급하는 강소기업으로, 나이키, 아디다스의 세계 3대 핵심 협력업체 중 한 곳이다. MBK파트너스는 연평균 10%에 가까운 전 세계 운동화 시장의 견조한 성장세는 물론, 전 세계 운동화 시장 점유율 45%를 차지하고 점유율을 확대해가고 있는 나이키 및 아디다스와 동진섬유와의 30년 이상 된 굳건한 협력 관계, 합성가죽보다는 섬유가 다양한 종류의 운동화에 지배적으로 사용되고 있는 추세 등을 긍정적으로 평가해 인수한 것으로 알려졌다."

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": prompt + sentence}
]

response = openai.ChatCompletion.create(
#response = openai.chat.completions.create(
            model="gpt-4", #3.5-turbo",
            messages=messages,
            temperature=0.0,  # 창의성을 조절하는 옵션
            #max_tokens=max_tokens,  # 답변의 최대 토큰 수 설정
           )

summary = response['choices'][0]['message']['content']

print(summary)

print('original: ', len(sentence))
print('summary: ', len(summary))

prompt = "다음 문장이 정치, 경제, 사회, 기술 중 어느 분야에 속하는지 알려줘. 정치면 P, 경제면 E, 사회면 S, 기술이면 T로 답변해줘."

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": prompt + sentence}
]

response = openai.ChatCompletion.create(
            model="gpt-4", #3.5-turbo",
            messages=messages,
            temperature=0.0,  # 창의성을 조절하는 옵션
            #max_tokens=max_tokens,  # 답변의 최대 토큰 수 설정
           )

topic = response['choices'][0]['message']['content']

print(topic)


