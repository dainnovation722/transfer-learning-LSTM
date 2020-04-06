#!/bin/bash

python main.py -m pre-train --gpu && \
python main.py -m transfer-learning --gpu && \
python main.py -m without-transfer-learning --gpu && \
python main.py -m score --gpu  
