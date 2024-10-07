#!/bin/bash

model='klue/roberta-base'
b_model=128
length=256
#pth_file='pths/klue_roberta-base_news_PEST_train_ksic_score_clean.tsv_topic_b128_e10_ml256_20240802_185749.pth'
pth_file='klue_roberta-base_news_PEST_ksic_score_clean.tsv_topic_b128_e20_ml256_20240805_202050.pth'

echo "python eval_classify2.py -te 'news_PEST_test_ksic_score_clean.tsv' -d /home/osung/data/korean/modu/json -m $model -b $b_model -v topic -l $length -t 16 -c /home/osung/Downloads/kisti_cert.crt -n 4 -f klue_roberta-base_news_PEST_ksic_score_clean.tsv_topic_code.csv $pth_file"

python eval_classify2.py -te 'news_PEST_train_ksic_score_clean.tsv' -d /home/osung/data/korean/modu/json -m $model -b $b_model -v topic -l $length -t 16 -c /home/osung/Downloads/kisti_cert.crt -n 4 -f klue_roberta-base_news_PEST_ksic_score_clean.tsv_topic_code.csv $pth_file
