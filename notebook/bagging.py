from os import path, listdir, makedirs
import sys
sys.path.append('../')
from utils.data_io import (
    read_data_from_dataset,
    ReccurentPredictingGenerator
)
import matplotlib.pyplot as plt
from keras.models import load_model
from tqdm import tqdm 
import numpy as np
from sklearn.metrics import mean_squared_error as mse

target = 'sru'
print(target)
write_out_dir = '../reports/bagging_100'
write_result_out_dir = path.join(write_out_dir, 'bagging', target)

# load dataset
data_dir_path = path.join('..', 'dataset', 'target', target)
X_train, y_train, X_test, y_test = \
    read_data_from_dataset(data_dir_path)

period = (len(y_train) + len(y_test)) // 30
RPG = ReccurentPredictingGenerator(X_test, batch_size=1, timesteps=period)
prediction = []


for path_model in tqdm(listdir(path.join(write_result_out_dir, 'model'))):
    file_path = path.join(write_result_out_dir, 'model', path_model)
    best_model = load_model(file_path)
    y_test_pred = best_model.predict_generator(RPG)
    prediction.append(y_test_pred)


prediction = np.array(prediction)
list_score = []
size_test = prediction.shape[1]
y_test = y_test[-size_test:]
for i_prediction in range(prediction.shape[0])[:1]:
    pred = np.mean(prediction[:i_prediction + 1], axis=0)
    accuracy = mse(y_test, pred.flatten())
    list_score.append(accuracy)

np.save('sru', prediction)

plt.rcParams['font.size'] = 25
plt.figure(figsize=(15, 7))
plt.plot(list_score)
plt.xlabel('the number of subsets / -')
plt.ylabel('MSE / -')
plt.savefig('bagging_sru')