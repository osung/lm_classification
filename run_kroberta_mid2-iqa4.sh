#!/bin/bash

python lm_classify.py --train train_mid2.tsv --test test_mid2.tsv -d ~/scratch/data/aihub/patent --model klue/roberta-large -b 384 -e 10 --num_labels 44 --max_length=256
