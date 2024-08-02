#!/bin/bash

python lm_classify.py --train train_summary_eenz-t5.tsv --test test_mid2.tsv --dir /home/osung/data/korean/patent --model 'monologg/koelectra-base-v3-discriminator' -b 128 -e 5 --num_labels 44 --max_length=512


