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


class ReccurentTrainingGenerator(Sequence):
    """ Reccurent レイヤーを訓練するためのデータgeneratorクラス """
    def _resetindices(self):
        """バッチとして出力するデータのインデックスを乱数で生成する """
        self.num_called = 0 # 同一のエポック内で __getitem__　メソッドが呼び出された回数
        
        all_idx = np.random.permutation(np.arange(self.num_samples))
        remain_idx = np.random.choice(np.arange(self.num_samples),
                                      size=(self.steps_per_epoch*self.batch_size-len(all_idx)),
                                      replace=False)
        self.indices = np.hstack([all_idx, remain_idx]).reshape(self.steps_per_epoch, self.batch_size)
        
    def __init__(self, x_set, y_set, batch_size, timesteps, delay):
        """
        x_set     : 説明変数 (データ点数×特徴量数)のNumPy配列
        y_set     : 目的変数 (データ点数×1)のNumPy配列
        batch_size: バッチサイズ
        timesteps : どの程度過去からデータをReccurent層に与えるか
        delay     : 目的変数をどの程度遅らせるか
        """
        self.x = np.array(x_set)
        self.y = np.array(y_set)
        self.batch_size = batch_size
        self.steps = timesteps
        self.delay = delay
        
        self.num_samples = len(self.x)-timesteps-delay+1
        self.steps_per_epoch = int(np.ceil( self.num_samples / float(batch_size)))
        
        self._resetindices()
        
    def __len__(self):
        """ 1エポックあたりのステップ数を返す """
        return self.steps_per_epoch
        
    def __getitem__(self, idx):
        """ データをバッチにまとめて出力する """
        indices_temp = self.indices[idx]
        
        batch_x = np.array([self.x[i:i+self.steps] for i in indices_temp])
        batch_y = self.y[indices_temp+self.steps+self.delay-1]
        
        if self.num_called==(self.steps_per_epoch-1):
            self._resetindices() # 1エポック内の全てのバッチを返すと、データをシャッフルする
        else:
            self.num_called += 1
        
        return batch_x, batch_y
    
    
class ReccurentPredictingGenerator(Sequence):
    """ Reccurent レイヤーで予測するためのデータgeneratorクラス """ 
    def __init__(self, x_set, batch_size, timesteps):
        """
        x_set     : 説明変数 (データ点数×特徴量数)のNumPy配列
        batch_size: バッチサイズ
        timesteps : どの程度過去からデータをReccurent層に与えるか
        """
        self.x = np.array(x_set)
        self.batch_size = batch_size
        self.steps = timesteps
        
        self.num_samples = len(self.x)-timesteps+1
        self.steps_per_epoch = int(np.floor(self.num_samples / float(batch_size)))
        
        self.idx_list = []
        
    def __len__(self):
        """ 1エポックあたりのステップ数を返す """
        return self.steps_per_epoch
        
    def __getitem__(self, idx):
        """ データをバッチにまとめて出力する """
        start_idx = idx*self.batch_size
        batch_x = [self.x[start_idx+i : start_idx+i+self.steps] for i in range(self.batch_size)]
        self.idx_list.append(start_idx)
        return np.array(batch_x)


def decompose_time_series(x):
    
    step = len(x) // 10
    best_score = np.inf
    print('decomposing time series data ・・・・・')
    for period in tqdm(range(1,step+1)):
        decompose_result = sm.tsa.seasonal_decompose(pd.Series(x), period=period, model='additive',extrapolate_trend='freq')
        score = np.sum(decompose_result.resid)
        if score < best_score:
            best_period = period
            best_score = score

    decompose_result = sm.tsa.seasonal_decompose(pd.Series(x), period=best_period, model='additive', extrapolate_trend='freq')

    x = {'trend': decompose_result.trend, 'period': decompose_result.seasonal, 'resid': decompose_result.resid}
    return x, best_period
