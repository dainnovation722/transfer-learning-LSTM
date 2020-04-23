import random
import keras
import argparse
import pickle
import json
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sys import exit
from os import path, getcwd, makedirs, environ, listdir
from time import time

from keras.models import load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
from utils.model import regressor
from utils.data_io import *
from utils.save import *
from utils.output import *
	
def seed_every_thing(seed=1234):
    environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_random_seed(seed)

def parse_arguments():
    parser = argparse.ArgumentParser(
                        description='Time-Series Regression by LSTM through transfer learning')
    # for dataset path
    parser.add_argument('--out-dir', '-o', default='result',
                        type=str, help='path for output directory')
    # for model
    parser.add_argument('--seed', type=int, default=1234,
                        help='seed value for random value, (default : 1234)')
    parser.add_argument('--train-ratio', default=0.8, type=float,
                        help='percentage of train data to be loaded (default : 0.9)')
    parser.add_argument('--time-window', default=1000, type=int, 
                        help='length of time to capture at once (default : 1000)')
    # for training
    parser.add_argument('--train-mode', '-m', default='pre-train', type=str, 
                        help='"pre-train", "transfer-learning", "without-transfer-learning", "score" (default : pre-train)')
    parser.add_argument('--gpu', action='store_true', 
                        help='whether to do calculations on gpu machines (default : True)')
    parser.add_argument('--nb-epochs', default=1, type=int, 
                        help='training epochs for the model (default : 1)')
    parser.add_argument('--nb-batch', default=20, type=int,
						help='number of batches in training (default : 20)')
	# for output 
    parser.add_argument('--train-verbose', default=1, type=int, 
                        help='whether to show the learning process (default : 1)')
    return parser.parse_args()

def save(model, y_test_time, y_pred_test_time,  write_result_out_dir):
	save_lr_curve(model).savefig(path.join(write_result_out_dir, 'learning_curve.png'))
	save_plot(y_test_time, y_pred_test_time).savefig(path.join(write_result_out_dir, 'prediction.png'))
	accuracy = mse(y_test_time,y_pred_test_time)
	with open(path.join(write_result_out_dir, 'log.txt'),'w') as f:
		f.write('accuracy : {:.6f}\n'.format(accuracy))
		f.write('='*65+'\n')       
		model.model.summary(print_fn=lambda x: f.write(x + '\n')) 

def load_dataset(data_dir_path, time_window, pre_train=None):
	X_train, y_train, X_test, y_test = read_data_from_dataset(data_dir_path)
	# in case of pre-training (no use for test, so concatenate train and test) 
	if pre_train:
		time_steps = time_window
		X_train = np.concatenate((X_train, X_test), axis=0)	
		y_train = np.concatenate((y_train, y_test), axis=0)
		X_train_time, y_train_time = generator(X_train, y_train, time_steps)
		return X_train_time, y_train_time
	# in case of transfer-learning
	if X_train.shape[0] > X_test.shape[0]:
		time_steps = time_window if X_test.shape[0]//2 > time_window else X_test.shape[0]//2 # Need to optimize time_window...
	else:
		time_steps = time_window if X_train.shape[0]//2 > time_window else X_train.shape[0]//2 
	X_train_time, y_train_time = generator(X_train, y_train, time_steps)
	X_test_time, y_test_time = generator(X_test, y_test, time_steps)
	return X_train_time, y_train_time, X_test_time, y_test_time

def make_callbacks(file_path):

	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.0001)
	model_checkpoint = ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)
	csv_logger = CSVLogger(path.join(path.dirname(file_path), 'epoch_log.csv'))
	callbacks=[reduce_lr, model_checkpoint, csv_logger]
	return callbacks

def main():	    

	# get args
    args = parse_arguments()

	# set seed
    seed_every_thing(args.seed)
	
	# make output base directory
    out_dir = args.out_dir
    write_out_dir = path.normpath(path.join(getcwd(), 'reports', out_dir))
    makedirs(write_out_dir, exist_ok=True)
	
	# save arguments
    with open(path.join(write_out_dir, 'params.json'), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)
	
    # set experiment parameters
    nb_batch = args.nb_batch
    nb_epochs = args.nb_epochs
    verbose = args.train_verbose
    time_window = args.time_window
    train_mode = args.train_mode
    train_ratio = args.train_ratio
    gpu = args.gpu

    print('-'*140)
    
    if train_mode == 'pre-train':
        
        for source in listdir('dataset/source'):  

            # skip source dataset without pickle file
            data_dir_path = path.join('dataset','source', source)
            if not path.exists(f'{data_dir_path}/X_train.pkl'):continue
            
            # make output directory
            write_result_out_dir = path.join(write_out_dir, train_mode, source)
            makedirs(write_result_out_dir, exist_ok=True)
            
            # load dataset
            X_train_time, y_train_time = \
                load_dataset(data_dir_path, time_window, pre_train=True)
            X_train_time, y_train_time, X_valid_time, y_valid_time = \
                split_dataset(X_train_time, y_train_time, ratio=train_ratio) 
            
            # construct the model 
            file_path = path.join(write_result_out_dir, 'best_model.hdf5')
            callbacks = make_callbacks(file_path)
            input_shape = (X_train_time.shape[1], X_train_time.shape[2]) # x_train.shape[2] is number of variable
            model = regressor(input_shape, gpu, write_result_out_dir)    
            
            # train the model
            print(f'\nSource dataset : {source}')
            print(f'\nSource dataset shape : {X_train_time.shape}')
            model.fit(X_train_time, y_train_time, X_valid_time, y_valid_time, nb_batch, nb_epochs, callbacks, verbose)
            
            # prediction and saving
            best_model = load_model(file_path)
            y_pred_valid_time = best_model.predict(X_valid_time) 
            save(model, y_valid_time, y_pred_valid_time, write_result_out_dir)
            
            keras.backend.clear_session()
            print('\n'*2+'-'*140+'\n'*2)
        

    elif train_mode == 'transfer-learning':
        
        for target in listdir('dataset/target'): 
        
            # skip target in the absence of pickle file
            if not path.exists(f'dataset/target/{target}/X_train.pkl'): continue		

            for source in listdir(f'{write_out_dir}/pre-train'): 	
                
                # skip target dataset if target is source 
                if target==source: continue
                
                # make output directory
                write_result_out_dir = path.join(write_out_dir, train_mode, target, source)
                makedirs(write_result_out_dir, exist_ok=True)
                    
                # load dataset
                data_dir_path = f'dataset/target/{target}' 
                X_train_time, y_train_time, X_test_time, y_test_time = \
                    load_dataset(data_dir_path, time_window)
                X_train_time, y_train_time, X_valid_time, y_valid_time = \
                    split_dataset(X_train_time, y_train_time, ratio=train_ratio) 

                # load pre-trained model
                pre_model = load_model(f'{write_out_dir}/pre-train/{source}/best_model.hdf5')
                
                # construct the model 
                file_path = path.join(write_result_out_dir, 'transferred_best_model.hdf5')
                callbacks = make_callbacks(file_path)                
                input_shape = (X_train_time.shape[1], X_train_time.shape[2]) # x_train.shape[2] is num of variable
                model = regressor(input_shape, gpu, write_result_out_dir, pre_model=pre_model)
        
                # train the model
                print(f'\nTarget dataset : {target}')
                print(f'\nSource dataset : {source}')
                print(f'\nTarget dataset shape : {X_train_time.shape}')
                model.fit(X_train_time, y_train_time, X_valid_time, y_valid_time, nb_batch, nb_epochs, callbacks, verbose)
                
                # prediction and saving
                best_model = load_model(file_path)
                y_pred_test_time = best_model.predict(X_test_time) 
                save(model, y_test_time, y_pred_test_time, write_result_out_dir)

                keras.backend.clear_session()
                print('\n'*2+'-'*140+'\n'*2)
    
    elif train_mode == 'without-transfer-learning':

        for target in listdir('dataset/target'): 
        
            # make output directory
            write_result_out_dir = path.join(write_out_dir, train_mode, target)
            makedirs(write_result_out_dir, exist_ok=True)

            # skip source dataset without pickle file
            data_dir_path = path.join('dataset', 'target', target)
            if not path.exists(f'{data_dir_path}/X_train.pkl'):continue
            
            # load dataset
            X_train_time, y_train_time = \
                load_dataset(data_dir_path, time_window, pre_train=True)
            X_train_time, y_train_time, X_valid_time, y_valid_time = \
                split_dataset(X_train_time, y_train_time, ratio=train_ratio) 
            
            # construct the model 
            file_path = path.join(write_result_out_dir, 'best_model.hdf5')
            callbacks = make_callbacks(file_path)
            input_shape = (X_train_time.shape[1], X_train_time.shape[2]) # x_train.shape[2] is number of variable
            model = regressor(input_shape, gpu, write_result_out_dir)
            
            # train the model
            print(f'\nTarget dataset : {target}')
            print(f'\nTarget dataset shape : {X_train_time.shape}')
            model.fit(X_train_time, y_train_time, X_valid_time, y_valid_time, nb_batch, nb_epochs, callbacks, verbose)
            
            # prediction and saving
            best_model = load_model(file_path)
            y_pred_valid_time = best_model.predict(X_valid_time) 
            save(model, y_valid_time, y_pred_valid_time, write_result_out_dir)
            
            keras.backend.clear_session()
            print('\n'*2+'-'*140+'\n'*2)

    # make results easy to see
    elif train_mode == 'score':
        metrics(write_out_dir)

if __name__ == '__main__':
	main()