import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from os import makedirs
import pickle

df = pd.read_table('traffic.txt',sep=',',header=None)
df.columns = range(df.shape[1])

selected_feature_idx = np.argsort(list(df.std(axis=0)))[::-1]

df[df.columns.tolist()]=MinMaxScaler().fit_transform(df[df.columns.tolist()])

y = df.iloc[:,selected_feature_idx[0]].values
X = df.iloc[:,selected_feature_idx[1:11]].values


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
for n_column in df.iloc[:,selected_feature_idx[1:11]]:
    plot_outlier(df.iloc[:,selected_feature_idx[1:11]][n_column],n_column)



