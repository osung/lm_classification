#!/bin/bash
echo "python news_classify_korscideberta2.py -tr train_data.tsv -d /home/osung/data/korean/kmaps_corpus -b 128 -v KSIC_m -l 256 -t 16 -c /home/osung/Downloads/kisti_cert.crt -e 10"
python news_classify_korscideberta2.py -tr train_data.tsv -d /home/osung/data/korean/kmaps_corpus -b 128 -v KSIC_m -l 256 -t 16 -c /home/osung/Downloads/kisti_cert.crt -e 10 --save_every_iter

echo "python news_classify_korscideberta2.py -tr train_data.tsv -d /home/osung/data/korean/kmaps_corpus -b 128 -v KSIC -l 256 -t 16 -c /home/osung/Downloads/kisti_cert.crt -e 10"
python news_classify_korscideberta2.py -tr train_data.tsv -d /home/osung/data/korean/kmaps_corpus -b 128 -v KSIC -l 256 -t 16 -c /home/osung/Downloads/kisti_cert.crt -e 10 --save_every_iter


