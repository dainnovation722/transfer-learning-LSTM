import tensorflow as tf
from keras import backend as K


def limit_gpu_memory():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
