
Transfer Learning LSTM for Time-Seires Regression
====   
## Description
This repository is a series of experiments on transfer learning for time-series data regression.

## Get Started
    $ git clone https://github.com/dainnovation722/transfer-learning-LSTM.git
    $ cd transfer-learning-LSTM
    $ conda create -n tl_env --file env_name.txt
    $ conda activate tl_env

## Experiments
    $ python main.py \
        --out-dir result \
        --seed 1234 \
        --train-ratio 0.8 \
        --time-window 1000 \
        --train-mode pre-train \
        --gpu \
        --nb-epochs 1 \
        --nb-batch 20 \
        --train-verbose 1 

### Easy execution
    $ ./train_all.sh

## Results

## Contribution
1. Fork it
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Create new Pull Reques

## Author
[dainnovation722](https://github.com/dainnovation722)
