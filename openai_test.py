import openai
import os

openai.api_key = "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"

_my_linux_ = 1

if _my_linux_  == 1:
   os.environ['CURL_CA_BUNDLE'] = '/home/osung/Downloads/kisti_cert.crt'

prompt = "Hello, how are you doing today?"

response = openai.Completion.create(
            engine="davinci", 
            prompt=prompt, 
            max_tokens=60, 
            n=1, 
            stop=None, 
            temperature=0.7,
        )

print(response.choices[0].text)

