import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def metrics(condition):
    targets = os.listdir(f'archives/{condition}/without-transfer-learning')
    sources = os.listdir(f'archives/{condition}/pre-train')   
    if not os.path.exists(f'archives/{condition}/results'): os.makedirs(f'archives/{condition}/results')
    pre_train = []
    df = []
    for target in targets:
        
        #転移学習無しの結果を取得
        with open(f'archives/{condition}/without-transfer-learning/{target}/log.txt','r') as f:
            pre_train.append(float(f.readlines()[0].split(':')[1].lstrip(' ')))

        
        #転移学習有りの結果を(1×sources)のrowで取得
        row = []
        for source in sources:
            with open(f'archives/{condition}/transfer-learning/to_{target}/from_{source}/log.txt', 'r') as f:
                row.append(float(f.readlines()[0].split(':')[1].lstrip(' ')))        
        df.append(row)
    
    tl = pd.DataFrame(np.array(df),columns=sources, index=targets)
    metrics_map = (1-tl.divide(pre_train, axis=0))*100

    sns.set(font_scale=2.0)
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(metrics_map.T, annot=True, fmt="1.1f", cmap="Blues", \
                cbar_kws={'label': 'mse decreasing rate after transfer [%]'},annot_kws={"size": 20})
    plt.xticks(rotation=20)
    plt.yticks(rotation=20)
    plt.tight_layout()
    ax.set_ylim(metrics_map.T.shape[0], 0)
    plt.show()
    plt.savefig(f'report/figure/{condition}.png')
    
    debutanizer = sorted(metrics_map.iloc[0,:].to_dict().items(), key=lambda x:x[1], reverse=True)
    debutanizer = np.array([ data[0] for data in debutanizer]).reshape(-1,1)
    sru = sorted(metrics_map.iloc[1,:].to_dict().items(), key=lambda x:x[1], reverse=True)
    sru = np.array([ data[0] for data in sru]).reshape(-1,1)
    rank = pd.DataFrame(np.concatenate([debutanizer,sru],axis=1),columns=['debutanizer','sru'])
    rank.to_csv(f'report/table/{condition}.csv', index=False)