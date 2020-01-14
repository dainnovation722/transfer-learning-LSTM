
Transfer Learning LSTM for Time-Seires Regression
====   
## Description
This repository is a series of experiments on transfer learning for time-series data regression.

## Requirement
    pip install requirements.txt
- keras
- matplotlib
- sklearn
- numpy 
- pickle
- dtw

## Usage
You can train the model by source dataset.
```
python main.py pre-train
```
You can train the model by target dataset through transfer learning.
```
python main.py transfer-learning
```
You can see dataset shape by this code.
```
python main.py data-info
```

## Install
    git clone git@github.com:dainnovation722/transfer-learning-LSTM.git
## Contribution
1. Fork it
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Create new Pull Reques

## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)

## Author

[dainnovation722](https://github.com/dainnovation722)
