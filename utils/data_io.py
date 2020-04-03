import pickle
import numpy as np

def read_data_from_dataset(data_dir_path:str) -> np.array:
    '''load train data and test data from the dataset'''
    data_list=[]
    for fname in ['X_train', 'y_train', 'X_test', 'y_test']:
        with open(f'{data_dir_path}/{fname}.pkl', 'rb') as f:
            data = pickle.load(f)
            data_list.append(data)
    return data_list[0], data_list[1], data_list[2], data_list[3]

def generator(X:np.array, y:np.array, time_steps:int) -> np.array:
    '''get time-series batch dataset'''    
    n_batches = X.shape[0] - time_steps - 1
    
    X_time = np.zeros((n_batches, time_steps, X.shape[1]))
    y_time = np.zeros((n_batches, 1))    
    for i in range(n_batches):
        X_time[i] = X[i:(i+time_steps),:]
        y_time[i] = y[i+time_steps]               
    return X_time, y_time

def split_dataset(X, y, ratio=0.8):
    '''split dataset to train data and valid data'''
    X_train = X[:int(X.shape[0]*ratio)]
    y_train = y[:int(y.shape[0]*ratio)]
    X_valid = X[int(X.shape[0]*ratio):]
    y_valid = y[int(y.shape[0]*ratio):]
    return X_train, y_train, X_valid, y_valid