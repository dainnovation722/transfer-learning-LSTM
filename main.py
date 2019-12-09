from keras.layers import Input, Dense, Activation, BatchNormalization
from keras.layers import CuDNNLSTM as LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 13
import warnings
warnings.simplefilter('ignore')

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
            data_list.append(data[:100])
    return data_list[0], data_list[1], data_list[2], data_list[3]

def generator(X:np.array, y:np.array) -> np.array:
    '''時系列バッチを取得する'''
    if X_train.shape[0] > X_test.shape[0]:
	    time_steps = 1000 if X_test.shape[0]//2 > 1000 else X_test.shape[0]//2 # ここのハイパラは調整する価値あり!
    else:
        time_steps = 1000 if X_train.shape[0]//2 > 1000 else X_train.shape[0]//2 # ここのハイパラは調整する価値あり!
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

	dense = TimeDistributed(Dense(50))(input_layer)

	lstm1 = LSTM(25, return_sequences=True)(dense)
	lstm1 = BatchNormalization()(lstm1)

	lstm2 = LSTM(50, return_sequences=True)(lstm1)
	lstm2 = BatchNormalization()(lstm2)

	lstm3 = LSTM(25, return_sequences=False)(lstm2)
	lstm3 = BatchNormalization()(lstm3)

	output_layer = Dense(1, activation='sigmoid')(lstm3)

	model = Model(inputs=input_layer, outputs=output_layer)

	if pre_model is not None:

		for i in range(2,len(model.layers)): 
			model.layers[i].set_weights(pre_model.layers[i].get_weights())
	
	model.compile(loss='mse', optimizer = Adam(), metrics=['accuracy'])
	return model

def save_fig(hist, y_pred_time):
	# 学習曲線の保存
	plt.figure(figsize=(18,5))
	plt.plot(hist.history['loss'])
	plt.plot(hist.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper right')
	plt.savefig(write_output_dir + 'learning_curve.png')
	# テストデータの予測結果の保存
	plt.figure(figsize=(18,5))
	plt.plot([ i for i in range(1, 1+len(y_pred_time))], y_pred_time, 'r',label="predicted")
	plt.plot([ i for i in range(1, 1+len(y_test_time))], y_test_time, 'b',label="measured", lw=1, alpha=0.3)
	plt.ylim(0,1)
	plt.legend(loc="best")
	plt.savefig(write_output_dir + 'prediction.png')
	
def train(pre_model=None):
	'''学習実行'''
	mini_batch_size = X_train_time.shape[0]//20
	
	start_time = time()

	input_shape = (X_train_time.shape[1], X_train_time.shape[2]) # x_train.shape[2] is num of variable
	model = build_model(input_shape, pre_model)

	if verbose == True: 
		model.summary()


	hist = model.fit(X_train_time, y_train_time, batch_size=mini_batch_size, epochs=nb_epochs,
		verbose=verbose, validation_data=(X_test_time, y_test_time), callbacks=callbacks)
	
	model = load_model(file_path)

	y_pred_time = model.predict(X_test_time)
	
	save_fig(hist,y_pred_time) #予測モデル図追加
	
	accuracy = mse(y_test_time,y_pred_time)
	
	duration = time()-start_time

	with open(write_output_dir+'log.txt','w') as f:
		f.write('duration : {:.3f}\n'.format(duration))
		f.write('accuracy : {:.3f}'.format(accuracy))

	keras.backend.clear_session()

if __name__ == '__main__':
	
	nb_epochs = 50
	verbose = 1
	targets = ['sru','debutanizer']
	print('='*140)
	if sys.argv[1] == 'pre-train':
		
		for source in os.listdir('dataset'):
			# sourceデータセットにpickleファイルがない場合は次のsourceデータセットへ
			if not os.path.exists(f'dataset/{source}/X_train.pkl'): continue
			# データセットの読み込み
			X_train, y_train, X_test, y_test = read_data_from_dataset(source)
			X_train_time, y_train_time = generator(X_train, y_train)
			X_test_time, y_test_time = generator(X_test, y_test)
			# 保存先フォルダー作成
			write_output_dir = f'pre-train/{source}/'
			if not os.path.exists(write_output_dir): os.makedirs(write_output_dir)
			file_path = write_output_dir + 'best_model.hdf5'
			print('Learning from '+source)
			print(f'Source Data Shape : {X_train_time.shape}')
			# 学習スケジューラー
			reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50,
					min_lr=0.0001)
			# モデルチェックポイント
			model_checkpoint = ModelCheckpoint(filepath=file_path, monitor='val_loss',
				save_best_only=True)
			# 各エポックの結果をcsvファイルに保存
			csv_logger = CSVLogger(write_output_dir+'epoch_log.csv')
			callbacks=[reduce_lr, model_checkpoint, csv_logger]

			train()

			print('\n'*2+'='*140+'\n'*2)			

	if sys.argv[1] == 'transfer-learning':
		
		for target in targets: 
			for source in os.listdir('dataset'): 
				# sourceデータセットにpickleファイルがない場合は次のsourceデータセットへ
				if not os.path.exists(f'dataset/{source}/X_train.pkl'): continue		
				if source in targets: continue
				# データセットの読み込み
				X_train, y_train, X_test, y_test = read_data_from_dataset(target)
				X_train_time, y_train_time = generator(X_train, y_train)
				X_test_time, y_test_time = generator(X_test, y_test)
				# 保存先フォルダー作成
				write_output_dir = f'transfer-learning/to_{target}/from_{source}/'
				if not os.path.exists(write_output_dir): os.makedirs(write_output_dir)
				file_path = write_output_dir+'transferred_best_model.hdf5'
				print('Tranfering from '+source+' to '+target)
				print(f'Target Data Shape : {X_train_time.shape}')
				# 事前学習済みモデルの読み込み
				pre_model = load_model(f'pre-train/{source}/best_model.hdf5')
				# 学習スケジューラー
				reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
						min_lr=0.0001)
				# モデルチェックポイント
				model_checkpoint = ModelCheckpoint(filepath=file_path, monitor='loss',
					save_best_only=True)
				# 各エポックの結果をcsvファイルに保存
				csv_logger = CSVLogger(write_output_dir+'epoch_log.csv')
				callbacks=[reduce_lr, model_checkpoint, csv_logger]

				train(pre_model)

				print('\n'*2+'='*140+'\n'*2)			

	if sys.argv[1] == 'data-info':
		print('='*30)
		print('dataset name	    shape')
		for data in os.listdir('dataset'):	
			if not os.path.exists(f'dataset/{data}/X_train.pkl'): continue
			X_train, y_train, X_test, y_test = read_data_from_dataset(data)
			print('-'*30)
			print('{:<15}'.format(data), end='')
			print('{:>15}'.format(str(X_train.shape)))
		print('-'*30)
			
	