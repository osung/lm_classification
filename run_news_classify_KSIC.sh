#!/bin/bash

#model='monologg/koelectra-base-v3-discriminator'
#b_model=64
#model='kisti/korscideberta'
#b_model=128
model='snunlp/KR-SBERT-V40K-klueNLI-augSTS'
b_model=128

echo "python news_classify.py -tr train_data.tsv -d /home/osung/data/korean/kmaps_corpus -m $model -b $b_model -v KSIC_m -l 256 -t 16 -c /home/osung/Downloads/kisti_cert.crt -e 10"
python news_classify.py -tr 'train_data.tsv' -d /home/osung/data/korean/kmaps_corpus -m $model -b $b_model -v KSIC_m -l 256 -t 16 -c /home/osung/Downloads/kisti_cert.crt -e 10 --save_every_iter

