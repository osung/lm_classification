#!/bin/bash

python lm_classify.py --train train_summary_eenz-t5.tsv --test test_mid2.tsv --dir /home01/hpc56a01/scratch/data/aihub/patent --model 'monologg/koelectra-base-v3-discriminator' -b 128 -e 5 --num_labels 44 --max_length=512


