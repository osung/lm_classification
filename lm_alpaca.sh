#!/bin/bash

python lm_classify2.py --train train_summary_psyche-kot5.tsv --test test_summary_psyche-kot5.tsv --dir /home01/hpc56a01/scratch/data/aihub/patent --model 'EleutherAI/polyglot-ko-1.3b' -b 16 -e 10 --max_length 1024 --add_pad_token
