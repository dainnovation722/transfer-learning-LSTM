#!/bin/bash

output="100_epochs"
epochs="100"
python main.py -m pre-train -o ${output} --gpu --nb-epochs $epochs && \
python main.py -m transfer-learning -o ${output} --gpu --nb-epochs $epochs && \
python main.py -m without-transfer-learning -o ${output} --gpu --nb-epochs $epochs && \
python main.py -m score -o ${output}