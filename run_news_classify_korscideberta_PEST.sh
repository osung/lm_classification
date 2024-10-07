#!/bin/bash
echo "python news_classify_korscideberta.py -tr train_PEST.tsv -d /home/osung/data/korean/kmaps_corpus -b 128 -v PEST -l 256 -t 16 -c /home/osung/Downloads/kisti_cert.crt -e 10 -n 4"
python news_classify_korscideberta.py -tr train_PEST.tsv -d /home/osung/data/korean/kmaps_corpus -b 128 -v PEST -l 256 -t 16 -c /home/osung/Downloads/kisti_cert.crt -e 10 -n 4 

