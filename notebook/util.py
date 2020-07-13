from os import listdir, path
import pickle

from dtw import dtw
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set(font_scale=1.4, font="Times New Roman")
# sns.set_style("ticks", {'font.family':'serif', 'font.serif':'Times New Roman'})
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


def dataset_idx_vs_improvement(feature_make, diff_cal=None):
    # load dataset
    source_path = '../dataset/source/'
    data_dict = {}
    for d_name in listdir(source_path):
        with open(path.join(source_path, d_name, 'y_test.pkl'), 'rb') as f:
            data = pickle.load(f)
        data_dict[d_name] = data
    
    target_path = '../dataset/target/'
    for d_name in listdir(target_path):
        with open(path.join(target_path, d_name, 'y_test.pkl'), 'rb') as f:
            data = pickle.load(f)
        data_dict[d_name] = data

    # feature make
    for key in data_dict.keys():
        data_dict[key] = feature_make(data_dict[key])

    # calculate difference
    dataset_index = pd.DataFrame(columns=['sru', 'debutanizer'], index=listdir(source_path))

    for key, value in data_dict.items():
        if key == 'sru' or key == 'debutanizer': continue
        if not diff_cal:
            print('please select how to calculate difference between dataset indices')
        if diff_cal == 'DTW':
            manhattan_distance = lambda x, y: np.abs(x - y)
            dataset_index.at[key, 'sru'] = dtw(data_dict['sru'], data_dict[key], dist=manhattan_distance)[0]
            dataset_index.at[key, 'debutanizer'] = dtw(data_dict['debutanizer'], data_dict[key], dist=manhattan_distance)[0]
        elif diff_cal == 'mse':
            dataset_index.at[key, 'sru'] = mse(data_dict['sru'], data_dict[key])
            dataset_index.at[key, 'debutanizer'] = mse(data_dict['debutanizer'], data_dict[key])
        
    # plot dataset_idx vs improvement
    mse_df = pd.read_csv('../reports/10_60_60_1/figure/mse.csv', index_col=0)
    improvement_df = pd.read_csv('../reports/10_60_60_1/figure/improvement.csv', index_col=0)
    print(dataset_index)
    print(improvement_df)
    for idx, target in enumerate(listdir(target_path)):
        plt.subplot(1, 2, idx + 1)
        x, y = [], []
        for source in listdir(source_path):
            x.append(dataset_index.at[source, target]*10**7)
            y.append(mse_df.at[target, source])
            
        plt.plot(x, y, 'b.', markersize=20, label='転移学習あり', color='blue')
        # for (i,j,k) in zip(x,y,listdir(source_path)):
            # plt.annotate(k, xy=(i, j))
        x_range = max(x) - min(x)
        x_min = min(x) - 0.1 * x_range
        x_max = max(x) + 0.1 * x_range
        plt.plot([x_min, x_max], [mse_df.at[target, 'base'] for _ in range(2)], linestyle='dashed', label='転移学習なし', color='black')
        plt.xlabel('ウェーブレット変換特徴量の非類似度 / -', fontweight='bold')
        plt.ylabel('MSE / -', fontweight='bold')
        target = target.capitalize() if target == 'debutanizer' else target.upper()
        plt.title(f'{target}')
        plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # plot dataset_idx rank vs improvement
    for idx, target in enumerate(listdir(target_path)):
        plt.subplot(1, 2, idx + 1)
        improvement_list = [improvement_df.at[target, source] for source in dataset_index[target].sort_values().keys().tolist()]  
        n = len(improvement_list)
        plt.plot(range(1, n + 1), improvement_list, 'b', label='with transfer')
        plt.plot(range(1, n + 1), [0 for _ in range(len(improvement_list))], 'r', label='without transfer', linestyle='dashed') 
        plt.xlabel('Feature Similarity Rank / -', fontweight='bold')
        plt.ylabel('Improvement / %', fontweight='bold')
        plt.yticks([i*50 for i in range(-3,4)])
        plt.legend(loc='best')
        target = target.capitalize() if target == 'debutanizer' else target.upper()
        plt.title(f'({"ab"[idx]}) {target}')
    plt.tight_layout()
    plt.show()
