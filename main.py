
from keras.models import load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
import keras
import matplotlib
import warnings
warnings.filterwarnings('ignore')
import argparse

from sklearn.metrics import mean_squared_error as mse

import numpy as np
import sys
import os
import pickle
from time import time

from utils.data_io import *
from utils.save import *
from utils.models import regressor
from utils.output import *
	
def save(model, y_test_time, y_pred_test_time,  write_output_dir):
	save_lr_curve(model).savefig(write_output_dir + 'learning_curve.png')
	save_plot(y_test_time, y_pred_test_time).savefig(write_output_dir + 'prediction.png')
	accuracy = mse(y_test_time,y_pred_test_time)
	with open(write_output_dir+'log.txt','w') as f:
		f.write('accuracy : {:.3f}\n'.format(accuracy))
		f.write('='*65+'\n')       
		model.model.summary(print_fn=lambda x: f.write(x + '\n')) #モデルアーキテクチャー

def load_dataset(data_dir_path, time_window, pre_train=None):
	X_train, y_train, X_test, y_test = read_data_from_dataset(data_dir_path)
	# pre-trainの場合(testデータは不要なのでtestデータをtrainデータにまとめる)
	if pre_train:
		time_steps = time_window
		X_train = np.concatenate((X_train, X_test), axis=0)	
		y_train = np.concatenate((y_train, y_test), axis=0)
		X_train_time, y_train_time = generator(X_train, y_train, time_steps)
		return X_train_time, y_train_time
	# transfer-learningの場合
	if X_train.shape[0] > X_test.shape[0]:
		time_steps = time_window if X_test.shape[0]//2 > time_window else X_test.shape[0]//2 # ここのハイパラは調整する価値あり!
	else:
		time_steps = time_window if X_train.shape[0]//2 > time_window else X_train.shape[0]//2 # ここのハイパラは調整する価値あり!
	X_train_time, y_train_time = generator(X_train, y_train, time_steps)
	X_test_time, y_test_time = generator(X_test, y_test, time_steps)
	return X_train_time, y_train_time, X_test_time, y_test_time

def make_callbacks(file_path, write_output_dir):
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.0001)
	model_checkpoint = ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)
	csv_logger = CSVLogger(write_output_dir+'epoch_log.csv')
	callbacks=[reduce_lr, model_checkpoint, csv_logger]
	return callbacks

def main():	    

	nb_batch = 20
	nb_epochs = 1
	verbose = 1 #学習途中の経過
	time_window = 1000
	print('='*140)

	parser = argparse.ArgumentParser() 
	parser.add_argument('-m', dest='mode', type=str, help='select train mode')
	parser.add_argument('-n', dest='name', type=str, help='name of result folder')
	args = parser.parse_args()
	mode = args.mode 
	condition = args.name

	if mode == 'pre-train':
		
		for source in os.listdir('dataset/source'):

			# pickleファイルがないsourceはスキップ
			data_dir_path = f'dataset/source/{source}' 
			if not os.path.exists(f'{data_dir_path}/X_train.pkl'):continue
			
			# 保存先フォルダー作成
			write_output_dir = f'archives/{condition}/pre-train/{source}/'
			if not os.path.exists(write_output_dir): os.makedirs(write_output_dir)
			file_path = write_output_dir + 'best_model.hdf5'
			
			# データセットの読み込み
			X_train_time, y_train_time = load_dataset(data_dir_path, time_window, pre_train=True)
			X_train_time, y_train_time, X_valid_time, y_valid_time = split_dataset(X_train_time, y_train_time, ratio=0.8) 
				
			# callbacks作成
			callbacks = make_callbacks(file_path, write_output_dir)
			
			print('\nLearning from '+source)
			print(f'\nSource Data Shape : {X_train_time.shape}\n')
			
			
			#モデル構築→学習→推論
			input_shape = (X_train_time.shape[1], X_train_time.shape[2]) # x_train.shape[2] is num of variable
			model = regressor(input_shape)
			model.fit(X_train_time, y_train_time, X_valid_time, y_valid_time, nb_batch, nb_epochs, callbacks, verbose)
			best_model = load_model(file_path)
			y_pred_valid_time = best_model.predict(X_valid_time) #予測には最高精度のモデルを使用
		
			#学習結果の保存
			save(model, y_valid_time, y_pred_valid_time, write_output_dir)
			
			keras.backend.clear_session()
			print('\n'*2+'='*140+'\n'*2)

	if mode == 'transfer-learning':
		
		for target in os.listdir('dataset/target'): 
			
			# pickleファイルがないtargetはスキップ
			if not os.path.exists(f'dataset/target/{target}/X_train.pkl'): continue		

			for source in os.listdir(f'archives/{condition}/pre-train'): 	
				
				#sourceとtargetが重複した際はスキップ
				if source==target: continue
				
				# 保存先フォルダー作成
				write_output_dir = f'archives/{condition}/transfer-learning/to_{target}/from_{source}/'
				if not os.path.exists(write_output_dir): os.makedirs(write_output_dir)
				file_path = write_output_dir+'transferred_best_model.hdf5'
				
				# データセットの読み込み
				data_dir_path = f'dataset/target/{target}' 
				X_train_time, y_train_time, X_test_time, y_test_time = load_dataset(data_dir_path, time_window)
				X_train_time, y_train_time, X_valid_time, y_valid_time = split_dataset(X_train_time, y_train_time, ratio=0.8) 

				# 事前学習済みモデルの読み込み
				pre_model = load_model(f'archives/{condition}/pre-train/{source}/best_model.hdf5')
				
				# callbacks作成
				callbacks = make_callbacks(file_path, write_output_dir)

				print('\nTranfering from '+source+' to '+target)
				print(f'\nTarget Data Shape : {X_train_time.shape}\n')
				
				#モデル構築→学習→推論
				input_shape = (X_train_time.shape[1], X_train_time.shape[2]) # x_train.shape[2] is num of variable
				model = regressor(input_shape, pre_model=pre_model)
				model.fit(X_train_time, y_train_time, X_valid_time, y_valid_time, nb_batch, nb_epochs, callbacks, verbose)
				best_model = load_model(file_path)
				y_pred_test_time = best_model.predict(X_test_time) #予測には最高精度のモデルを使用

				#学習結果の保存
				save(model, y_test_time, y_pred_test_time, write_output_dir)

				keras.backend.clear_session()
				print('\n'*2+'='*140+'\n'*2)
	
	if mode == 'without-transfer-learning':

		for target in os.listdir('dataset/target'): 
			
			# pickleファイルがないtargetはスキップ
			if not os.path.exists(f'dataset/target/{target}/X_train.pkl'): continue				
			
			# 保存先フォルダー作成
			write_output_dir = f'archives/{condition}/without-transfer-learning/{target}/'
			if not os.path.exists(write_output_dir): os.makedirs(write_output_dir)
			file_path = write_output_dir+'transferred_best_model.hdf5'
			
			# データセットの読み込み
			data_dir_path = f'dataset/target/{target}' 
			X_train_time, y_train_time, X_test_time, y_test_time = load_dataset(data_dir_path, time_window)
			X_train_time, y_train_time, X_valid_time, y_valid_time = split_dataset(X_train_time, y_train_time, ratio=0.8) 

			# callbacks作成
			callbacks = make_callbacks(file_path, write_output_dir)
		
			print('\nLearning from '+target)
			print(f'\nSource Data Shape : {X_train_time.shape}\n')
				
			#モデル構築→学習→推論
			input_shape = (X_train_time.shape[1], X_train_time.shape[2]) # x_train.shape[2] is num of variable
			model = regressor(input_shape)
			model.fit(X_train_time, y_train_time, X_valid_time, y_valid_time, nb_batch, nb_epochs, callbacks, verbose)
			best_model = load_model(file_path)
			y_pred_test_time = best_model.predict(X_test_time) #予測には最高精度のモデルを使用

			#学習結果の保存
			save(model, y_test_time, y_pred_test_time, write_output_dir)

			keras.backend.clear_session()
			print('\n'*2+'='*140+'\n'*2)

	if mode == 'output-results':
			metrics(condition)

if __name__ == '__main__':
	main()
	

			

			
	