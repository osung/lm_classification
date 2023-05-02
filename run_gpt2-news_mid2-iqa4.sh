#!/bin/bash
python lm_classify.py --train train_mid2.tsv --test test_mid2.tsv --dir /home01/hpc56a01/scratch/data/aihub/patent --model 'jihae/kogpt2news' -b 1024 -e 10 --num_labels 44 --max_length=4881
