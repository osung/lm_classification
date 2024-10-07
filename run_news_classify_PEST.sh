#!/bin/bash

#model='monologg/koelectra-base-v3-discriminator'
#b_model=64
#model='kisti/korscideberta'
#b_model=128
#model='snunlp/KR-SBERT-V40K-klueNLI-augSTS'
#b_model=128
model='klue/roberta-base'
b_model=128
length=256

echo "python lm_classify_PEST.py -tr news_PEST_ksic_score_summary_no_dup_train.tsv -te news_PEST_ksic_score_summary_no_dup_test.tsv -d /home/osung/data/korean/modu/json -m $model -b $b_model -v topic -l $length -t 16 -c /home/osung/Downloads/kisti_cert.crt -e 20 -n 4 --save_every_iter"


python lm_classify_PEST.py -tr news_PEST_ksic_score_summary_no_dup_train.tsv -te news_PEST_ksic_score_summary_no_dup_test.tsv -d /home/osung/data/korean/modu/json -m $model -b $b_model -v topic -l $length -t 16 -c /home/osung/Downloads/kisti_cert.crt -e 20 -n 4 --save_every_iter

