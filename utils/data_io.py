import pickle
import numpy as np
def read_data_from_dataset(data_dir_path:str) -> np.array:
    '''データセットから学習データとテストデータを読み込む'''
    data_list=[]
    for fname in ['X_train', 'y_train', 'X_test', 'y_test']:
        with open(f'{data_dir_path}/{fname}.pkl', 'rb') as f:
            data = pickle.load(f)
            data_list.append(data)
    return data_list[0], data_list[1], data_list[2], data_list[3]


def generator(X:np.array, y:np.array, time_steps:int) -> np.array:
    '''時系列バッチを取得する'''    
    n_batches = X.shape[0] - time_steps - 1
    X_time = np.zeros((n_batches, time_steps, X.shape[1]))
    y_time = np.zeros((n_batches, 1))    
    for i in range(n_batches):
        X_time[i] = X[i:(i+time_steps),:]
        y_time[i] = y[i+time_steps]               
    return X_time, y_time

