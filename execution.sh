#!/bin/bash

# python main.py pre-train && python main.py transfer-learning && python main.py without-transfer-learning

#事例データ

condition="transfer_in_only_lstm"
n_iteration=1


for n_iter in `seq 1 $n_iteration`
do
    python main.py -m pre-train -n "${condition}_${n_iter}" && \
    python main.py -m transfer-learning -n "${condition}_${n_iter}" && \
    python main.py -m without-transfer-learning -n "${condition}_${n_iter}" && \
    python main.py -m output-results -n "${condition}_${n_iter}"
done
