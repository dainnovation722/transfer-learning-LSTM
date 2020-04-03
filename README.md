
Transfer Learning LSTM for Time-Seires Regression
====   
## Description
This repository is a series of experiments on transfer learning for time-series data regression.

## Getting Started
    $ git clone https://github.com/dainnovation722/transfer-learning-LSTM.git
    $ cd transfer-learning-LSTM
    $ python -v venv venv
    $ source venv/bin/activate
    $ pip install -r requirements.txt


## Experiments

### **1. Pre-train**  
Pre-train the model by source dataset
```
$ python main.py pre-train
```
### **2. Transfer learning**  
Train the model by target dataset through pre-trained mode
```
$ python main.py transfer-learning
```

## Contribution
1. Fork it
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Create new Pull Reques


## Author
[dainnovation722](https://github.com/dainnovation722)
