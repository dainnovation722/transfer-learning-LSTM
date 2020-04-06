from os import makedirs, path, listdir
from sys import exit
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def metrics(condition):

    # make output base directory
    base_out_dir = path.join('reports', condition, 'figure')
    
    makedirs(base_out_dir, exist_ok=True)
    source_dir = path.join('reports', condition, 'pre-train')
    target_dir = path.join('reports', condition, 'transfer-learning')
    
    no_tl, tl = [], []
    for target in listdir(target_dir):
        # fetch results without transfer learning
        with open(path.join('reports', condition, 'without-transfer-learning', target, 'log.txt'), 'r') as f:
            no_tl.append(float(f.readlines()[0].split(':')[1].lstrip(' ')))
        # fetch results as row(1×sources) with transfer learning
        row = []
        for source in listdir(source_dir):
            with open(path.join(target_dir, target, source, 'log.txt'), 'r') as f:
                row.append(float(f.readlines()[0].split(':')[1].lstrip(' ')))        
        tl.append(row)
    
    tl = pd.DataFrame(np.array(tl),columns=listdir(source_dir), index=listdir(target_dir))
    metrics_map = (1-tl.divide(no_tl, axis=0))*100
    
    sns.set(font_scale=2.0)
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(metrics_map.T, annot=True, fmt="1.1f", cmap="Blues", \
                cbar_kws={'label': 'mse decreasing rate after transfer [%]'},annot_kws={"size": 20})
    plt.xticks(rotation=20)
    plt.yticks(rotation=20)
    plt.tight_layout()
    ax.set_ylim(metrics_map.T.shape[0], 0)
    plt.savefig(path.join(base_out_dir, 'metrics.png'))
    
    debutanizer = sorted(metrics_map.iloc[0,:].to_dict().items(), key=lambda x:x[1], reverse=True)
    debutanizer = np.array([ data[0] for data in debutanizer]).reshape(-1,1)
    sru = sorted(metrics_map.iloc[1,:].to_dict().items(), key=lambda x:x[1], reverse=True)
    sru = np.array([ data[0] for data in sru]).reshape(-1,1)
    rank = pd.DataFrame(np.concatenate([debutanizer,sru],axis=1),columns=['debutanizer','sru'])
    rank.to_csv(path.join(base_out_dir, 'metrics.csv'), index=False)