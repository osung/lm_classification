#!/bin/bash

directory='/home/osung/data/korean/patent/mids/mid_train'

for file in "$directory"/??_train_summary_psyche-kot5.tsv; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")

        echo "python lm_classify2.py -tr $filename -d /home/osung/data/korean/patent/mids/mid_train -m monologg/koelectra-base-v3-discriminator -b 192 -v scode -l 256 -t 16 -c /home/osung/Downloads/kisti_cert.crt -e 10"
        python lm_classify2.py -tr $filename -d /home/osung/data/korean/patent/mids/mid_train -m klue/roberta-large -b 128 -v scode -l 256 -t 16 -c /home/osung/Downloads/kisti_cert.crt -e 10
    fi
done

#python lm_classify2.py -tr 01_train_summary_psyche-kot5.tsv -d /home/osung/data/korean/patent/mids/mid_train -m monologg/koelectra-base-v3-discriminator -b 192  -v scode -l 256 -t 16 -c /home/osung/Downloads/kisti_cert.crt -e 1

