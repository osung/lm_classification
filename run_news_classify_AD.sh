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


echo "python lm_classify_ad.py -tr ad_total.tsv -d /home/osung/data/korean/commercial/csv -m $model -b $b_model -v ad -l $length -t 16 -c /home/osung/Downloads/kisti_cert.crt -e 10 -n 2" 

python lm_classify_ad.py -tr ad_total.tsv -d /home/osung/data/korean/commercial/csv -m $model -b $b_model -v ad -l $length -t 16 -c /home/osung/Downloads/kisti_cert.crt -e 10 -n 2 
