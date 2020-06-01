import pickle
import numpy as np
from keras.utils import Sequence
from tqdm import tqdm
from statsmodels import api as sm
import pandas as pd


def read_data_from_dataset(data_dir_path: str):
    """load train data and test data from the dataset

    Args:
        data_dir_path (str): path for the dataset we want to load

    Returns:
        tuple: tuple which contains X_train, y_train, X_test and y_test in order
    """
    data_list = []
    for fname in ['X_train', 'y_train', 'X_test', 'y_test']:
        with open(f'{data_dir_path}/{fname}.pkl', 'rb') as f:
            data = pickle.load(f)
            data_list.append(data)
    return tuple(data_list)


def generator(X: np.array, y: np.array, time_steps: int):
    """get time-series batch dataset

    Args:
        X (np.array): data for explanatory variables
        y (np.array): data for target variable
        time_steps (int): length of time series to consider during learning

    Returns:
        X_time (np.array): preprocessed data for explanatory variables
        y_time (np.array): preprocessed data for target variable

    """
    n_batches = X.shape[0] - time_steps - 1
    
    X_time = np.zeros((n_batches, time_steps, X.shape[1]))
    y_time = np.zeros((n_batches, 1))
    for i in range(n_batches):
        X_time[i] = X[i:(i + time_steps), :]
        y_time[i] = y[i + time_steps]
    return X_time, y_time


def split_dataset(X: np.array, y: np.array, ratio=0.8):
    """split dataset to train data and valid data in deep learning

    Args:
        X (np.array): data for explanatory variables
        y (np.array): data for target variable
        ratio (float, optional): ratio of train data and valid data. Defaults to 0.8.

    Returns:
        tuple: tuple which contains X_train, y_train, X_valid and y_valid in order
    """
    '''split dataset to train data and valid data'''
    X_train = X[:int(X.shape[0] * ratio)]
    y_train = y[:int(y.shape[0] * ratio)]
    X_valid = X[int(X.shape[0] * ratio):]
    y_valid = y[int(y.shape[0] * ratio):]
    dataset = tuple([X_train, y_train, X_valid, y_valid])

    return dataset
