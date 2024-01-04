#!/bin/bash

directory='/home/osung/data/korean/patent/mids/mid_train'

echo "python news_classify.py -tr kmpas_corpus_0104_new.tsv -d /home/osung/data/korean/kmaps_corpus -m monologg/koelectra-base-v3-discriminator -b 64 -v PEST -l 256 -t 16 -c /home/osung/Downloads/kisti_cert.crt -e 10 -n 4"
python news_classify.py -tr 'kmaps_corpus_0104_new.tsv' -d /home/osung/data/korean/kmaps_corpus -m monologg/koelectra-base-v3-discriminator -b 64 -v PEST -l 256 -t 16 -c /home/osung/Downloads/kisti_cert.crt -e 10 -n 4


