#!/bin/bash

model='kisti/korscideberta'
b_model=128

### PEST ###
#target='PEST'
#code_csv='saves/kisti_korscideberta_train_data.tsv_PEST_code.csv'
#pth_file="saves/kisti_korscideberta_train_data.tsv_PEST_b128_e10_ml256_20240109_220452.pth"


### KSIC_m ###
#target='KSIC_m'
#code_csv='saves/kisti_korscideberta_train_data.tsv_KSIC_m_code.csv'
#pth_file="saves/kisti_korscideberta_train_data.tsv_KSIC_m_b128_e10_ml256_20240110_093433.pth"


### KSIC ###
target='KSIC'
code_csv='saves/kisti_korscideberta_train_data.tsv_KSIC_code.csv'
pth_file="saves/kisti_korscideberta_train_data.tsv_KSIC_b128_e10_ml256_20240110_115423.pth"



### GO! GO! GO! ###
    echo "python eval_classify.py -te test_data.tsv -d /home/osung/data/korean/kmaps_corpus -m $model -b $b_model -v ${target} -l 256 -t 16 -c /home/osung/Downloads/kisti_cert.crt -C $code_csv $pth_file"
    python eval_classify.py -te test_data.tsv -d /home/osung/data/korean/kmaps_corpus -m $model -b $b_model -v ${target} -l 256 -t 16 -c /home/osung/Downloads/kisti_cert.crt -C $code_csv $pth_file 
