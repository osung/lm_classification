#!/bin/bash

for (( i=1; i<63; i++ )); do
#filename=$(printf "pths/monologg_koelectra-base-v3-discriminator_%02d_train_summary_*.pth" $i)
    filename=$(printf "pths/klue_roberta-large_%02d_train_summary_*.pth" $i)

    files=( $filename )

#echo ${#files[@]}

    if [ -f ${files[0]} ]; then
        echo "python eval_classify2.py -te $(printf "%02d_test_summary_psyche-kot5.tsv" $i) -d /home/osung/data/korean/patent/mids/mid_test -m klue/roberta-large -b 192 -v scode -f $(printf "/home/osung/data/korean/patent/mids/mid_train/%02d_scode.csv" $i) -c /home/osung/Downloads/kisti_cert.crt -t 16 -l 512 ${files[0]}"

        python eval_classify2.py -te $(printf "%02d_test_summary_psyche-kot5.tsv" $i) -d /home/osung/data/korean/patent/mids/mid_test -m klue/roberta-large -b 192 -v scode -f $(printf "/home/osung/data/korean/patent/mids/mid_train/%02d_scode.csv" $i) -c /home/osung/Downloads/kisti_cert.crt -t 16 -l 512 ${files[0]}

    fi

done

