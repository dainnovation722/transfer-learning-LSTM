from keras.layers import Input, Dense, Activation, BatchNormalization
from keras.layers import CuDNNLSTM as LSTM
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import keras
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 18

from sklearn.metrics import mean_squared_error as mse

import numpy as np
import sys
import os
import pickle
from time import time

def read_data_from_dataset(dname:str) -> np.array:
    '''データセットから学習データとテストデータを読み込む'''
    data_list=[]
    for fname in ['X_train', 'y_train', 'X_test', 'y_test']:
        with open(f'dataset/{dname}/{fname}.pkl', 'rb') as f:
            data = pickle.load(f)
            data_list.append(data)
    return data_list[0], data_list[1], data_list[2], data_list[3]

def generator(X:np.array, y:np.array) -> np.array:
    '''時系列バッチを取得する'''
    time_steps = 1000 if X_test.shape[0]//2 > 1000 else X_test.shape[0]//2 # ここのハイパラは調整する価値あり!
    n_batches = X.shape[0] - time_steps - 1
    X_time = np.zeros((n_batches, time_steps, X.shape[1]))
    y_time = np.zeros((n_batches, 1))    
    for i in range(n_batches):
        X_time[i] = X[i:(i+time_steps),:]
        y_time[i] = y[i+time_steps]               
    return X_time, y_time


def build_model(input_shape:tuple, pre_model=None) -> list:
	'''学習モデルを構築'''
	input_layer = Input(input_shape)

	lstm1 = LSTM(25, return_sequences=True)(input_layer)
	lstm1 = BatchNormalization()(lstm1)

	lstm2 = LSTM(50, return_sequences=True)(lstm1)
	lstm2 = BatchNormalization()(lstm2)

	lstm3 = LSTM(25, return_sequences=False)(lstm2)
	lstm3 = BatchNormalization()(lstm3)

	output_layer = Dense(1, activation='sigmoid')(lstm3)

	model = Model(inputs=input_layer, outputs=output_layer)

	if pre_model is not None:

		for i in range(len(model.layers)): #range(1,len(model.layers))またはrange(2,len(model.layers))
			model.layers[i].set_weights(pre_model.layers[i].get_weights())

	model.compile(loss='mse', optimizer = Adam(), metrics=['accuracy'])

	return model

def train(pre_model=None):
	'''学習実行'''
	mini_batch_size = X_train_time.shape[0]//10
	
	start_time = time()

	input_shape = (X_train_time.shape[1], X_train_time.shape[2]) # x_train.shape[2] is num of variable
	model = build_model(input_shape, pre_model)

	if verbose == True: 
		model.summary()

	
	hist = model.fit(X_train_time, y_train_time, batch_size=mini_batch_size, epochs=nb_epochs,
		verbose=verbose, validation_data=(X_test_time, y_test_time), callbacks=callbacks)

	
	model = load_model(file_path)

	y_pred_time = model.predict(X_test_time)
	accuracy = mse(y_test_time,y_pred_time)
	
	duration = time.time()-start_time

	with open(write_output_dir+'log.txt','wb') as f:
		f.write('duration : {:.3f}'.format(duration))
		f.write('accuracy : {:.3f}'.format(accuracy))

	keras.backend.clear_session()

if __name__ == '__main__':

	nb_epochs = 30
	verbose = 1
	
	if sys.argv[1] == 'pre-train':
		
		for source in os.listdir('dataset'):
			# sourceデータセットにpickleファイルがない場合は次のsourceデータセットへ
			if not os.path.exists(f'dataset/{source}/X_train.pkl'): continue
			# データセットの読み込み
			X_train, y_train, X_test, y_test = read_data_from_dataset(source)
			X_train_time, y_train_time = generator(X_train, y_train)
			X_test_time, y_test_time = generator(X_test, y_test)
			# 保存先フォルダー作成
			write_output_dir = 'pre-train/{source}/'
			if not os.path.exists(write_output_dir): os.makedirs(write_output_dir)
			file_path = write_output_dir + 'best_model.hdf5'
			print('Learning from '+source)
			# 学習打ち切り
			early_stopping = EarlyStopping(patience=5)
			# 学習スケジューラー
			reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
					min_lr=0.0001)
			# モデルチェックポイント
			model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
				save_best_only=True)
			callbacks=[early_stopping, reduce_lr,model_checkpoint]

			train()

	if sys.argv[1] == 'transfer-learning':
		for source in os.listdir('dataset'): 
			# sourceデータセットにpickleファイルがない場合は次のsourceデータセットへ
			if not os.path.exists(f'dataset/{source}/X_train.pkl'): continue
			for target in ['sru','debutanizer']: 
				if target == source: continue
				# データセットの読み込み
				X_train, y_train, X_test, y_test = read_data_from_dataset(source)
				X_train_time, y_train_time = generator(X_train, y_train)
				X_test_time, y_test_time = generator(X_test, y_test)
				# 保存先フォルダー作成
				write_output_dir = f'transfer-learning/to_{target}/from_{source}/'
				if not os.path.exists(write_output_dir): os.makedirs(write_output_dir)
				file_path = write_output_dir+'transferred_best_model.hdf5'
				print('Tranfering from '+source+' to '+target)
				# 事前学習済みモデルの読み込み
				pre_model = load_model(f'pre-train/{source}/best_model.hdf5')
				# 学習打ち切り
				early_stopping = EarlyStopping(patience=5)
				# 学習スケジューラー
				reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
					min_lr=0.0001)
				# モデルチェックポイント
				model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
					save_best_only=True)
				callbacks=[early_stopping, reduce_lr,model_checkpoint]

				train(pre_model)