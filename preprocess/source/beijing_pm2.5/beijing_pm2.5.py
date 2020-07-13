import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from os import makedirs
import pickle

df = pd.read_csv('PRSA_data_2010.1.1-2014.12.31.csv')

# filling or removing for NaN
df.interpolate(inplace=True)
df.dropna(axis=0, inplace=True)
df.drop(['No','year','month','day','hour','cbwd'], axis=1, inplace=True)

# scaling
df[df.columns.tolist()]=MinMaxScaler().fit_transform(df[df.columns.tolist()])

# save dataset
target = 'pm2.5'
y = df[target].values
X = df.drop([target], axis=1).values

data_dic = {'X_train':X[:int(X.shape[0]*0.8)],
            'y_train':y[:int(X.shape[0]*0.8)],
            'X_test':X[int(y.shape[0]*0.8):],
            'y_test':y[int(y.shape[0]*0.8):]}
for key,value in data_dic.items():
    with open(f'{key}.pkl','wb') as f:
        pickle.dump(value, f)

makedirs('feature', exist_ok=True)
def plot_outlier(ts, n_column, ewm_span=100, threshold=3.0):
    assert type(ts) == pd.Series
    fig, ax = plt.subplots(figsize=(15,5))
    ewm_mean = ts.ewm(span=ewm_span).mean()  
    ewm_std = ts.ewm(span=ewm_span).std()  
    ax.plot(ts, label='original')
    ax.plot(ewm_mean, label='ewma')

    # plot data which deviate from range during mean ±  3 * std as outlier 
    ax.fill_between(ts.index,
                    ewm_mean - ewm_std * threshold,
                    ewm_mean + ewm_std * threshold,
                    alpha=0.2)
    outlier = ts[(ts - ewm_mean).abs() > ewm_std * threshold]
    ax.scatter(outlier.index, outlier, label='outlier')
    ax.legend()
    plt.title(n_column)
    plt.savefig(f'feature/{n_column}.png')

# save feature plot 
for n_column in df.columns:
    plot_outlier(df[n_column],n_column)