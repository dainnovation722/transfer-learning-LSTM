#!/bin/bash

python main.py -m pre-train -o "${output}" --gpu && \
python main.py -m transfer-learning -o "${output}" --gpu && \
python main.py -m without-transfer-learning -o "${output}" --gpu && \
python main.py -m score -o "${output}" --gpu  
