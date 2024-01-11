#!/bin/bash

model='monologg/koelectra-base-v3-discriminator'
b_model=64
code_csv='saves/monologg_koelectra-base-v3-discriminator_train_PEST.tsv_PEST_code.csv'



for i in {0..9}; do
    pth_file="saves/monologg_koelectra-base-v3-discriminator_train_PEST.tsv_PEST_b64_e10_ml256_20240108_071953.pth_0${i}"

#model='kisti/korscideberta'
#b_model=128
#model='snunlp/KR-SBERT-V40K-klueNLI-augSTS'
#b_model=128

#model='klue/roberta-base'
#b_model=128
#code_csv='saves/klue_roberta-base_train_PEST.tsv_PEST_code.csv'
#pth_file='saves/klue_roberta-base_train_PEST.tsv_PEST_b128_e10_ml256_20240108_164959.pth'

#model='snunlp/KR-SBERT-V40K-klueNLI-augSTS'
#b_model=128
#code_csv='saves/snunlp_KR-SBERT-V40K-klueNLI-augSTS_train_PEST.tsv_PEST_code.csv'

#for i in {0..9}; do
 
#    pth_file="saves/snunlp_KR-SBERT-V40K-klueNLI-augSTS_train_PEST.tsv_PEST_b128_e10_ml256_20240108_130953.pth_0${i}"

    echo "python eval_classify.py -te test_PEST.tsv -d /home/osung/data/korean/kmaps_corpus -m $model -b $b_model -v PEST -l 256 -t 16 -c /home/osung/Downloads/kisti_cert.crt -C $code_csv $pth_file"
    python eval_classify.py -te 'test_PEST.tsv' -d /home/osung/data/korean/kmaps_corpus -m $model -b $b_model -v PEST -l 256 -t 16 -c /home/osung/Downloads/kisti_cert.crt -C $code_csv $pth_file
done
