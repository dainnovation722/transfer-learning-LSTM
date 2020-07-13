from keras.layers import Input, Dense, BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import Adam
from keras import initializers, regularizers
import numpy as np


def build_model(input_shape: tuple,
                gpu,
                write_result_out_dir,
                pre_model=None,
                freeze=False,
                noise=None,
                verbose=True,
                savefig=True):

    if gpu:
        from keras.layers import CuDNNLSTM as LSTM
    else:
        from keras.layers import LSTM

    # construct the model
    input_layer = Input(input_shape)
    
    if noise:
        noise_input = GaussianNoise(np.sqrt(noise))(input_layer)
        dense = TimeDistributed(
            Dense(
                10,
                kernel_regularizer=regularizers.l2(0.01),
                kernel_initializer=initializers.glorot_uniform(seed=0),
                bias_initializer=initializers.Zeros()
            )
        )(noise_input)

    else:
        dense = TimeDistributed(
            Dense(
                10,
                kernel_regularizer=regularizers.l2(0.01),
                kernel_initializer=initializers.glorot_uniform(seed=0),
                bias_initializer=initializers.Zeros()
            )
        )(input_layer)

    lstm1 = LSTM(
        60,
        return_sequences=True,
        kernel_regularizer=regularizers.l2(0.01),
        kernel_initializer=initializers.glorot_uniform(seed=0),
        recurrent_initializer=initializers.Orthogonal(seed=0),
        bias_initializer=initializers.Zeros()
    )(dense)
    lstm1 = BatchNormalization()(lstm1)

    lstm2 = LSTM(
        60,
        return_sequences=False,
        kernel_regularizer=regularizers.l2(0.01),
        kernel_initializer=initializers.glorot_uniform(seed=0),
        recurrent_initializer=initializers.Orthogonal(seed=0),
        bias_initializer=initializers.Zeros()
    )(lstm1)
    lstm2 = BatchNormalization()(lstm2)

    output_layer = Dense(
        1,
        activation='sigmoid',
        kernel_regularizer=regularizers.l2(0.01),
        kernel_initializer=initializers.glorot_uniform(seed=0),
        bias_initializer=initializers.Zeros()
    )(lstm2)

    model = Model(inputs=input_layer, outputs=output_layer)
    if savefig:
        plot_model(model, to_file=f'{write_result_out_dir}/architecture.png', show_shapes=True, show_layer_names=False)
    
    # transfer weights from pre-trained model
    if pre_model:
        for i in range(2, len(model.layers) - 1):
            model.layers[i].set_weights(pre_model.layers[i].get_weights())
            if freeze: 
                model.layers[i].trainable = False
            
    model.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])
    if verbose: print(model.summary())

    return model
    
