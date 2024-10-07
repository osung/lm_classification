#!/bin/bash

model='klue/roberta-base'
b_model=128
length=256
#pth_file='klue_roberta-base_ad_total.tsv_ad_b128_e10_ml256_20240814_135702.pth'
#pth_file='klue_roberta-base_ad_train.tsv_ad_b128_e10_ml256_20240814_114734.pth'
#pth_file='klue_roberta-base_ad_summary_train.tsv_ad_b128_e10_ml256_20240923_092800.pth'
pth_file='klue_roberta-base_ad_summary.tsv_ad_b128_e10_ml256_20240923_101517.pth'

echo "python eval_classify_ad.py -te ad_test.tsv -d /home/osung/data/korean/commercial/csv -m $model -b $b_model -v ad -l $length -t 16 -c /home/osung/Downloads/kisti_cert.crt -n 2 -f klue_roberta-base_ad_train.tsv_ad_code.csv $pth_file"

python eval_classify_ad.py -te ad_test.tsv -d /home/osung/data/korean/commercial/csv -m $model -b $b_model -v ad -l $length -t 16 -c /home/osung/Downloads/kisti_cert.crt -n 2 -f klue_roberta-base_ad_train.tsv_ad_code.csv $pth_file
