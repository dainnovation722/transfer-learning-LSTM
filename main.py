import random
import argparse
import json
import math
from os import path, getcwd, makedirs, environ, listdir

import tensorflow as tf
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from recombinator.optimal_block_length import optimal_block_length
from recombinator.block_bootstrap import circular_block_bootstrap

from utils.model import build_model
from utils.data_io import (
    read_data_from_dataset,
    ReccurentTrainingGenerator,
    ReccurentPredictingGenerator
)
from utils.save import save_lr_curve, save_prediction_plot, save_yy_plot, save_mse
from utils.device import limit_gpu_memory


def parse_arguments():
    ap = argparse.ArgumentParser(
        description='Time-Series Regression by LSTM through transfer learning')
    # for dataset path
    ap.add_argument('--out-dir', '-o', default='result',
                    type=str, help='path for output directory')
    # for model
    ap.add_argument('--seed', type=int, default=1234,
                    help='seed value for random value, (default : 1234)')
    ap.add_argument('--train-ratio', default=0.8, type=float,
                    help='percentage of train data to be loaded (default : 0.9)')
    ap.add_argument('--time-window', default=1000, type=int,
                    help='length of time to capture at once (default : 1000)')
    # for training
    ap.add_argument('--train-mode', '-m', default='pre-train', type=str,
                    help='"pre-train", "transfer-learning", "without-transfer-learning", \
                            "bagging", "noise-injection", "score" (default : pre-train)')
    ap.add_argument('--gpu', action='store_true',
                    help='whether to do calculations on gpu machines (default : False)')
    ap.add_argument('--nb-epochs', '-e', default=1, type=int,
                    help='training epochs for the model (default : 1)')
    ap.add_argument('--nb-batch', default=20, type=int,
                    help='number of batches in training (default : 20)')
    ap.add_argument('--nb-subset', default=10, type=int,
                    help='number of data subset in bootstrapping (default : 10)')
    ap.add_argument('--noise-var', default=0.0001, type=float,
                    help='variance of noise in noise injection (default : 0.0001)')
    ap.add_argument('--valid-ratio', default=0.2, type=float,
                    help='ratio of validation data in train data (default : 0.2)')
    ap.add_argument('--freeze', action='store_true', 
                    help='whether to freeze transferred weights in transfer learning (default : False)')
    # for output
    ap.add_argument('--train-verbose', default=1, type=int,
                    help='whether to show the learning process (default : 1)')
    args = vars(ap.parse_args())
    return args
    

def seed_every_thing(seed=1234):
    environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_random_seed(seed)


def save_arguments(args, out_dir):
    path_arguments = path.join(out_dir, 'params.json')
    if not path.exists(path_arguments):
        with open(path_arguments, mode="w") as f:
            json.dump(args, f, indent=4)


def make_callbacks(file_path, save_csv=True):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.000001)
    model_checkpoint = ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)
    if not save_csv:
        return [reduce_lr, model_checkpoint]
    csv_logger = CSVLogger(path.join(path.dirname(file_path), 'epoch_log.csv'))
    return [reduce_lr, model_checkpoint, csv_logger]
 

def main():

    # make analysis environment
    limit_gpu_memory()
    args = parse_arguments()
    seed_every_thing(args["seed"])
    write_out_dir = path.normpath(path.join(getcwd(), 'reports', args["out_dir"]))
    makedirs(write_out_dir, exist_ok=True)
    
    print('-' * 140)
    
    if args["train_mode"] == 'pre-train':
        
        for source in listdir('dataset/source'):

            # skip source dataset without pickle file
            data_dir_path = path.join('dataset', 'source', source)
            if not path.exists(f'{data_dir_path}/X_train.pkl'): continue
            
            # make output directory
            write_result_out_dir = path.join(write_out_dir, args["train_mode"], source)
            makedirs(write_result_out_dir, exist_ok=True)
            
            # load dataset
            X_train, y_train, X_test, y_test = \
                read_data_from_dataset(data_dir_path)
            period = (len(y_train) + len(y_test)) // 30
            X_train = np.concatenate((X_train, X_test), axis=0)  # no need for test data when pre-training
            y_train = np.concatenate((y_train, y_test), axis=0)  # no need for test data when pre-training
            X_train, X_valid, y_train, y_valid =  \
                train_test_split(X_train, y_train, test_size=args["valid_ratio"], shuffle=False)
            print(f'\nSource dataset : {source}')
            print(f'\nX_train : {X_train.shape[0]}')
            print(f'\nX_valid : {X_valid.shape[0]}')
            
            # construct the model
            file_path = path.join(write_result_out_dir, 'best_model.hdf5')
            callbacks = make_callbacks(file_path)
            input_shape = (period, X_train.shape[1])
            model = build_model(input_shape, args["gpu"], write_result_out_dir)
            
            # train the model
            bsize = len(y_train) // args["nb_batch"]
            RTG = ReccurentTrainingGenerator(X_train, y_train, batch_size=bsize, timesteps=period, delay=1)
            RVG = ReccurentTrainingGenerator(X_valid, y_valid, batch_size=bsize, timesteps=period, delay=1)
            H = model.fit_generator(RTG, validation_data=RVG, epochs=args["nb_epochs"], verbose=1, callbacks=callbacks)
            save_lr_curve(H, write_result_out_dir)

            # clear memory up
            keras.backend.clear_session()
            save_arguments(args, write_result_out_dir)
            print('\n' * 2 + '-' * 140 + '\n' * 2)
    
    elif args["train_mode"] == 'transfer-learning':
        
        for target in listdir('dataset/target'):
        
            # skip target in the absence of pickle file
            if not path.exists(f'dataset/target/{target}/X_train.pkl'): continue

            for source in listdir(f'{write_out_dir}/pre-train'):
                
                # make output directory
                write_result_out_dir = path.join(write_out_dir, args["train_mode"], target, source)
                pre_model_path = f'{write_out_dir}/pre-train/{source}/best_model.hdf5'
                if not path.exists(pre_model_path): continue
                makedirs(write_result_out_dir, exist_ok=True)
                    
                # load dataset
                data_dir_path = f'dataset/target/{target}'
                X_train, y_train, X_test, y_test = \
                    read_data_from_dataset(data_dir_path)
                period = (len(y_train) + len(y_test)) // 30
                X_train, X_valid, y_train, y_valid = \
                    train_test_split(X_train, y_train, test_size=args["valid_ratio"], shuffle=False)
                print(f'\nTarget dataset : {target}')
                print(f'\nX_train : {X_train.shape[0]}')
                print(f'\nX_valid : {X_valid.shape[0]}')
                print(f'\nX_test : {X_test.shape[0]}')
                
                # construct the model
                pre_model = load_model(pre_model_path)
                file_path = path.join(write_result_out_dir, 'transferred_best_model.hdf5')
                callbacks = make_callbacks(file_path)
                input_shape = (period, X_train.shape[1])
                model = build_model(input_shape, args["gpu"], write_result_out_dir, pre_model=pre_model, freeze=args["freeze"])
        
                # train the model
                bsize = len(y_train) // args["nb_batch"]
                RTG = ReccurentTrainingGenerator(X_train, y_train, batch_size=bsize, timesteps=period, delay=1)
                RVG = ReccurentTrainingGenerator(X_valid, y_valid, batch_size=bsize, timesteps=period, delay=1)
                H = model.fit_generator(RTG, validation_data=RVG, epochs=args["nb_epochs"], verbose=1, callbacks=callbacks)
                save_lr_curve(H, write_result_out_dir)
                
                # prediction
                best_model = load_model(file_path)
                RPG = ReccurentPredictingGenerator(X_test, batch_size=1, timesteps=period)
                y_test_pred = best_model.predict_generator(RPG)

                # save log for the model
                y_test = y_test[-len(y_test_pred):]
                save_prediction_plot(y_test, y_test_pred, write_result_out_dir)
                save_yy_plot(y_test, y_test_pred, write_result_out_dir)
                mse_score = save_mse(y_test, y_test_pred, write_result_out_dir, model=best_model)
                args["mse"] = mse_score
                save_arguments(args, write_result_out_dir)
                keras.backend.clear_session()
                print('\n' * 2 + '-' * 140 + '\n' * 2)
    
    elif args["train_mode"] == 'without-transfer-learning':

        for target in listdir('dataset/target'):
        
            # make output directory
            write_result_out_dir = path.join(write_out_dir, args["train_mode"], target)
            makedirs(write_result_out_dir, exist_ok=True)

            # load dataset
            data_dir_path = path.join('dataset', 'target', target)
            X_train, y_train, X_test, y_test = \
                read_data_from_dataset(data_dir_path)
            period = (len(y_train) + len(y_test)) // 30
            X_train, X_valid, y_train, y_valid =  \
                train_test_split(X_train, y_train, test_size=args["valid_ratio"], shuffle=False)
            print(f'\nTarget dataset : {target}')
            print(f'\nX_train : {X_train.shape[0]}')
            print(f'\nX_valid : {X_valid.shape[0]}')
            print(f'\nX_test : {X_test.shape[0]}')
            
            # construct the model
            file_path = path.join(write_result_out_dir, 'best_model.hdf5')
            callbacks = make_callbacks(file_path)
            input_shape = (period, X_train.shape[1])
            model = build_model(input_shape, args["gpu"], write_result_out_dir)
            
            # train the model
            bsize = len(y_train) // args["nb_batch"]
            RTG = ReccurentTrainingGenerator(X_train, y_train, batch_size=bsize, timesteps=period, delay=1)
            RVG = ReccurentTrainingGenerator(X_valid, y_valid, batch_size=bsize, timesteps=period, delay=1)
            H = model.fit_generator(RTG, validation_data=RVG, epochs=args["nb_epochs"], verbose=1, callbacks=callbacks)
            save_lr_curve(H, write_result_out_dir)

            # prediction
            best_model = load_model(file_path)
            RPG = ReccurentPredictingGenerator(X_test, batch_size=1, timesteps=period)
            y_test_pred = best_model.predict_generator(RPG)

            # save log for the model
            y_test = y_test[-len(y_test_pred):]
            save_prediction_plot(y_test, y_test_pred, write_result_out_dir)
            save_yy_plot(y_test, y_test_pred, write_result_out_dir)
            mse_score = save_mse(y_test, y_test_pred, write_result_out_dir, model=best_model)
            args["mse"] = mse_score
            save_arguments(args, write_result_out_dir)

            # clear memory up
            keras.backend.clear_session()
            print('\n' * 2 + '-' * 140 + '\n' * 2)

    elif args["train_mode"] == 'bagging':
    
        for target in listdir('dataset/target'):
            
            # make output directory
            write_result_out_dir = path.join(write_out_dir, args["train_mode"], target)
            makedirs(write_result_out_dir, exist_ok=True)

            # load dataset
            data_dir_path = path.join('dataset', 'target', target)
            X_train, y_train, X_test, y_test = \
                read_data_from_dataset(data_dir_path)
            period = (len(y_train) + len(y_test)) // 30

            # make subsets
            b_star = optimal_block_length(y_train)
            b_star_cb = math.ceil(b_star[0].b_star_cb)
            print(f'optimal block length for circular bootstrap = {b_star_cb}')
            subsets_y_train = circular_block_bootstrap(y_train, block_length=b_star_cb,
                                                       replications=args["nb_subset"], replace=True)
            subsets_X_train = []
            for i in range(X_train.shape[1]):
                np.random.seed(0)
                X_cb = circular_block_bootstrap(X_train[:, i], block_length=b_star_cb,
                                                replications=args["nb_subset"], replace=True)
                subsets_X_train.append(X_cb)
            subsets_X_train = np.array(subsets_X_train)
            subsets_X_train = subsets_X_train.transpose(1, 2, 0)

            # train the model for each subset
            model_dir = path.join(write_result_out_dir, 'model')
            makedirs(model_dir, exist_ok=True)
            for i_subset, (i_X_train, i_y_train) in enumerate(zip(subsets_X_train, subsets_y_train)):
                
                i_X_train, i_X_valid, i_y_train, i_y_valid = \
                    train_test_split(i_X_train, i_y_train, test_size=args["valid_ratio"], shuffle=False)
                
                # construct the model
                file_path = path.join(model_dir, f'best_model_{i_subset}.hdf5')
                callbacks = make_callbacks(file_path, save_csv=False)
                input_shape = (period, i_X_train.shape[1])  # x_train.shape[2] is number of variable
                model = build_model(input_shape, args["gpu"], write_result_out_dir, savefig=False)

                # train the model
                bsize = len(i_y_train) // args["nb_batch"]
                RTG = ReccurentTrainingGenerator(i_X_train, i_y_train, batch_size=bsize, timesteps=period, delay=1)
                RVG = ReccurentTrainingGenerator(i_X_valid, i_y_valid, batch_size=bsize, timesteps=period, delay=1)
                H = model.fit_generator(RTG, validation_data=RVG, epochs=args["nb_epochs"], verbose=1, callbacks=callbacks)
            
            keras.backend.clear_session()
            print('\n' * 2 + '-' * 140 + '\n' * 2)

    elif args["train_mode"] == 'noise-injection':

        for target in listdir('dataset/target'):
            
            # make output directory
            write_result_out_dir = path.join(write_out_dir, args["train_mode"], target)
            makedirs(write_result_out_dir, exist_ok=True)

            # load dataset
            data_dir_path = path.join('dataset', 'target', target)
            X_train, y_train, X_test, y_test = \
                read_data_from_dataset(data_dir_path)
            period = (len(y_train) + len(y_test)) // 30
            X_train, X_valid, y_train, y_valid =  \
                train_test_split(X_train, y_train, test_size=args["valid_ratio"], shuffle=False)
            print(f'\nTarget dataset : {target}')
            print(f'\nX_train : {X_train.shape}')
            print(f'\nX_valid : {X_valid.shape}')
            print(f'\nX_test : {X_test.shape[0]}')

            # construct the model
            file_path = path.join(write_result_out_dir, 'best_model.hdf5')
            callbacks = make_callbacks(file_path)
            input_shape = (period, X_train.shape[1])
            model = build_model(input_shape, args["gpu"], write_result_out_dir, noise=args["noise_var"])

            # train the model
            bsize = len(y_train) // args["nb_batch"]
            RTG = ReccurentTrainingGenerator(X_train, y_train, batch_size=bsize, timesteps=period, delay=1)
            RVG = ReccurentTrainingGenerator(X_valid, y_valid, batch_size=bsize, timesteps=period, delay=1)
            H = model.fit_generator(RTG, validation_data=RVG, epochs=args["nb_epochs"], verbose=1, callbacks=callbacks)
            save_lr_curve(H, write_result_out_dir)

            # prediction
            best_model = load_model(file_path)
            RPG = ReccurentPredictingGenerator(X_test, batch_size=1, timesteps=period)
            y_test_pred = best_model.predict_generator(RPG)

            # save log for the model
            y_test = y_test[-len(y_test_pred):]
            save_prediction_plot(y_test, y_test_pred, write_result_out_dir)
            save_yy_plot(y_test, y_test_pred, write_result_out_dir)
            mse_score = save_mse(y_test, y_test_pred, write_result_out_dir, model=best_model)
            args["mse"] = mse_score
            save_arguments(args, write_result_out_dir)

            # clear memory up
            keras.backend.clear_session()
            print('\n' * 2 + '-' * 140 + '\n' * 2)


if __name__ == '__main__':
    main()
