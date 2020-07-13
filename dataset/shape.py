import pickle
from os import listdir, path

for dname in listdir('source'):
    print(dname)
    n_features = 0
    n_samples = 0
    for fname in listdir(path.join('source',dname)):
        with open(path.join('source',dname,fname),'rb') as f:
            data = pickle.load(f)
        # print(f'{fname.replace("pkl","")} : {data.shape}')
        if fname == "X_train.pkl" or fname == "X_test.pkl":
            n_samples += data.shape[0]
            n_features = data.shape[1]
    print(f'Samples : {n_samples}')
    print(f'Features : {n_features}')
    print('-'*30)

for dname in listdir('target'):
    print(dname)
    n_features = 0
    n_samples = 0
    for fname in listdir(path.join('target',dname)):
        with open(path.join('target',dname,fname),'rb') as f:
            data = pickle.load(f)
        # print(f'{fname.replace("pkl","")} : {data.shape}')
        if fname == "X_train.pkl" or fname == "X_test.pkl":
            n_samples += data.shape[0]
            n_features = data.shape[1]
    print(f'Total : {n_samples}')
    print(f'Features : {n_features}')
    print('-'*30)