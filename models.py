"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses

# Doesn't work for some reason
import tensorflow.compat.v1 as tf
# But this does
# import tensorflow._api.v2.compat.v1 as tf
from tensorflow.compat.v1.keras import datasets, layers, models, losses

tf.disable_v2_behavior()

X_SHAPE = (28, 28, 1)

class Model(object):
    def __init__(self):
        self.model = models.Sequential()

        self.model.add(layers.Reshape((28, 28, 1)))
        # 4 Convolutional Layers
        self.model.add(layers.Conv2D(
            filters=32,
            kernel_size=3,
            input_shape=X_SHAPE
        ))
        self.model.add(layers.Conv2D(
            filters=64,
            kernel_size=4,
        ))
        self.model.add(layers.Conv2D(
            filters=128,
            kernel_size=3,
        ))
        self.model.add(layers.Conv2D(
            filters=256,
            kernel_size=3,
        ))

        # Flatten Convolutional Layers result
        self.model.add(layers.Flatten())

        # 2 Fully Connected Layers
        self.model.add(layers.Dense(
            units=1024,
        ))
        self.model.add(layers.Dense(
            units=1024,
        ))

        # Output Layer
        self.model.add(layers.Dense(
            units=10,
            activation='softmax'
        ))

        # self.model.compile(
        #     optimizer='SGD',
        #     loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        #     metrics=['accuracy']
        # )