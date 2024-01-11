#!/bin/bash

###### KOELECTRA ######
#model='monologg/koelectra-base-v3-discriminator'
#b_model=64

## KSIC_M ##
#code_csv='saves/monologg_koelectra-base-v3-discriminator_train_data.tsv_KSIC_m_code.csv'
#pth_file='saves/monologg_koelectra-base-v3-discriminator_train_data.tsv_KSIC_m_b64_e10_ml256_20240108_145346.pth'
#target='KSIC_m'

## KSIC ##
#code_csv='saves/monologg_koelectra-base-v3-discriminator_train_data.tsv_KSIC_code.csv'
#pth_file='saves/monologg_koelectra-base-v3-discriminator_train_data.tsv_KSIC_b64_e10_ml256_20240108_091344.pth'
#target='KSIC'




###### ROBERTA ######
#model='klue/roberta-base'
#b_model=128

## KSIC_M ##
#code_csv='saves/klue_roberta-base_train_data.tsv_KSIC_m_code.csv'
#pth_file='saves/klue_roberta-base_train_data.tsv_KSIC_m_b128_e10_ml256_20240108_182926.pth'
#target='KSIC_m'

## KSIC ##
#code_csv='saves/klue_roberta-base_train_data.tsv_KSIC_code.csv'
#pth_file='saves/klue_roberta-base_train_data.tsv_KSIC_b128_e10_ml256_20240108_200844.pth'
#target='KSIC'



###### SBERT ######
#model='snunlp/KR-SBERT-V40K-klueNLI-augSTS'
#b_model=128

## KSIC_m ##
#code_csv='saves/snunlp_KR-SBERT-V40K-klueNLI-augSTS_train_data.tsv_KSIC_m_code.csv'
#pth_file="saves/snunlp_KR-SBERT-V40K-klueNLI-augSTS_train_data.tsv_KSIC_m_b128_e10_ml256_20240108_214804.pth"
#target='KSIC_m'


## KSIC ##
#code_csv='saves/snunlp_KR-SBERT-V40K-klueNLI-augSTS_train_data.tsv_KSIC_code.csv'
#pth_file="saves/snunlp_KR-SBERT-V40K-klueNLI-augSTS_train_data.tsv_KSIC_b128_e10_ml256_20240108_232752.pth"
#target='KSIC'




###### KORSCIDEBERTA ######
model='kisti/korscideberta'
b_model=128


## KSIC_m ##
#code_csv='saves/kisti_korscideberta_train_data.tsv_KSIC_m_code.csv'
#pth_file="saves/kisti_korscideberta_train_data.tsv_KSIC_m_b128_e10_ml256_20240109_132620.pth"
#target='KSIC_m'


## KSIC ##
code_csv='saves/kisti_korscideberta_train_data.tsv_KSIC_code.csv'
pth_file="saves/kisti_korscideberta_train_data.tsv_KSIC_b128_e10_ml256_20240109_194349.pth"
target='KSIC'




###### GO! GO! GO! ######
    echo "python eval_classify.py -te test_data.tsv -d /home/osung/data/korean/kmaps_corpus -m $model -b $b_model -v $target -l 256 -t 16 -c /home/osung/Downloads/kisti_cert.crt -C $code_csv $pth_file"
    python eval_classify.py -te test_data.tsv -d /home/osung/data/korean/kmaps_corpus -m $model -b $b_model -v $target -l 256 -t 16 -c /home/osung/Downloads/kisti_cert.crt -C $code_csv $pth_file



