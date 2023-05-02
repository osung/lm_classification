#!/bin/bash

python lm_classify.py --train train_mid.tsv --test test_mid.tsv --dir /home01/hpc56a01/scratch/data/aihub/patent --model 'krevas/finance-koelectra-base-discriminator' -b 2048 -e 10 --num_labels 44

