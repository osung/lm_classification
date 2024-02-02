#!/bin/bash

#model='monologg/koelectra-base-v3-discriminator'
#pth_file="saves/monologg_koelectra-base-v3-discriminator_modu_news_pest_reduced_train.tsv_code_b128_e10_ml256_20240129_093915.pth"

#pth_file="saves/monologg_koelectra-base-v3-discriminator_modu_news_pest_reduced_train.tsv_code_b256_e10_ml256_20240122_092821.pth_00"

#model='snunlp/KR-SBERT-V40K-klueNLI-augSTS'
#pth_file='saves/snunlp_KR-SBERT-V40K-klueNLI-augSTS_modu_news_pest_reduced_train.tsv_code_b128_e10_ml256_20240125_085058.pth'

model='klue/roberta-base'
pth_file='saves/klue_roberta-base_modu_news_pest_reduced_train.tsv_code_b256_e10_ml256_20240118_112955.pth'

b_model=128

    echo "python eval_classify.py -te modu_news_pest_reduced_test.tsv -d /home/osung/data/korean/modu/modu_newspaper -m $model -b $b_model -l 256 -t 16 -c /home/osung/Downloads/kisti_cert.crt $pth_file"

    python eval_classify.py -te modu_news_pest_reduced_test.tsv -d /home/osung/data/korean/modu/modu_newspaper -m $model -b $b_model -v code -n 4 -l 256 -t 16 -c /home/osung/Downloads/kisti_cert.crt $pth_file -w modu_news_pest_reduced_test_incorrect.tsv
