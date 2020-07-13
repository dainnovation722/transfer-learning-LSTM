from os import path

from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.size"] = 13


def save_lr_curve(H, out_dir: str, f_name=None):
    """save learning curve in deep learning

    Args:
        model : trained model (keras)
        out_dir (str): directory path for saving
    """
    f_name = 'learning_curve' if not f_name else f_name
    plt.figure(figsize=(18, 5))
    plt.rcParams["font.size"] = 18
    plt.plot(H.history['loss'])
    plt.plot(H.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper right')
    plt.savefig(path.join(out_dir, f'{f_name}.png'))


def save_prediction_plot(y_test_time: np.array, y_pred_test_time: np.array, out_dir: str):
    """save prediction plot for tareget varibale

    Args:
        y_test_time (np.array): observed data for target variable
        y_pred_test_time (np.array): predicted data for target variable
        out_dir (str): directory path for saving
    """
    plt.figure(figsize=(18, 5))
    plt.rcParams["font.size"] = 18
    plt.plot([i for i in range(1, 1 + len(y_pred_test_time))], y_pred_test_time, 'r', label="predicted")
    plt.plot([i for i in range(1, 1 + len(y_test_time))], y_test_time, 'b', label="measured", lw=1, alpha=0.3)
    plt.ylim(0, 1)
    plt.xlim(0, len(y_test_time))
    plt.ylabel('Value')
    plt.xlabel('Time')
    plt.legend(loc="best")
    plt.savefig(path.join(out_dir, 'prediction.png'))


def save_yy_plot(y_test_time: np.array, y_pred_test_time: np.array, out_dir: str):
    """save yy plot for target variable

    Args:
        y_test_time (np.array): observed data for target variable
        y_pred_test_time (np.array): predicted data for target variable
        out_dir (str): directory path for saving
    """
    plt.figure(figsize=(10, 10))
    plt.rcParams["font.size"] = 18
    plt.plot(y_test_time, y_pred_test_time, 'b.')
    diagonal = np.linspace(0, 1, 10000)
    plt.plot(diagonal, diagonal, 'r-')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.savefig(path.join(out_dir, 'yy_plot.png'))


def save_yy_plot(y_test_time: np.array, y_pred_test_time: np.array, out_dir: str):
    """save yy plot for target variable

    Args:
        y_test_time (np.array): observed data for target variable
        y_pred_test_time (np.array): predicted data for target variable
        out_dir (str): directory path for saving
    """
    plt.figure(figsize=(10, 10))
    plt.rcParams["font.size"] = 18
    plt.plot(y_test_time, y_pred_test_time, 'b.')
    diagonal = np.linspace(0, 1, 10000)
    plt.plot(diagonal, diagonal, 'r-')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.savefig(path.join(out_dir, 'yy_plot.png'))


def save_mse(y_test_time: np.array, y_pred_test_time: np.array, out_dir: str, model=None):
    """save mean squared error for tareget variable

    Args:
        y_test_time (np.array): observed data for target variable
        y_pred_test_time (np.array): predicted data for target variable
        out_dir (str): directory path for saving
        model : trained model (keras)
    """
    accuracy = mse(y_test_time, y_pred_test_time)
    with open(path.join(out_dir, 'log.txt'), 'w') as f:
        f.write('accuracy : {:.6f}\n'.format(accuracy))
        f.write('=' * 65 + '\n')
        if model:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
    return accuracy
