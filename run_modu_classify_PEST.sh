#!/bin/bash

model='monologg/koelectra-base-v3-discriminator'
b_model=128
#model='kisti/korscideberta'
#b_model=128
#model='snunlp/KR-SBERT-V40K-klueNLI-augSTS'
#b_model=128
#model='klue/roberta-base'
#b_model=128

echo "python news_classify.py -tr modu_news_pest_reduced_train.tsv -d /home/osung/data/korean/modu/modu_newspaper -m $model -b $b_model -v code -l 256 -t 16 -c /home/osung/Downloads/kisti_cert.crt -e 10 -n 4 --save_every_iter"

python news_classify.py -tr modu_news_pest_reduced_train.tsv -d /home/osung/data/korean/modu/modu_newspaper -m $model -b $b_model -v code -l 256 -t 16 -c /home/osung/Downloads/kisti_cert.crt -e 10 -n 4 --save_every_iter
