import random
import numpy as np

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import datasets, utils
tf.disable_v2_behavior()

class MNISTDataLoader():
    def __init__(self):
        (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data('./datasets')
        self.X_train, self.X_test = MNISTDataLoader._preprocess_X(X_train), MNISTDataLoader._preprocess_X(X_test)
        self.Y_train, self.Y_test = MNISTDataLoader._preprocess_Y(Y_train), MNISTDataLoader._preprocess_Y(Y_test)
    
    @staticmethod
    def _preprocess_X(X_data):
        return X_data[..., np.newaxis]/255.0
    
    @staticmethod
    def _preprocess_Y(Y_data, num_classes=10):
        # return utils.to_categorical(Y_data, num_classes)
        return Y_data
