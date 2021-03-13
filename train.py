from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

import os
import sys
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from models import Model
from datasets import MNISTDataLoader

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

SEED = 12
DATASET_RATIO = 0.2
LEARNING_RATE_0 = 0.1
LEARNING_RATE_1 = 0.01
EPOCH_RATE_CHANGE = 50
EPOCHS = 60
BATCH_SIZE = 8 # 120

def create_dataset(dataset, batch_size):
    n = len(dataset)
    
    i = 0
    new_dataset = []
    while i*batch_size < n:
        X_agg = []
        Y_agg = []
        for k in range(i*batch_size, min(n, (i+1)*batch_size)):
            X_agg.append(dataset[k][0])
            Y_agg.append(dataset[k][1])
        new_dataset.append((X_agg, Y_agg))
        i += 1
    
    return new_dataset

np.random.seed(SEED)

mnist = MNISTDataLoader()

rand_idx_train = np.random.choice(len(mnist.X_train), size=int(DATASET_RATIO*len(mnist.X_train)))
rand_idx_test = np.random.choice(len(mnist.X_test), size=int(DATASET_RATIO*len(mnist.X_test)))

X_train = mnist.X_train[rand_idx_train]
Y_train = mnist.Y_train[rand_idx_train]

# Join datasets
train_dataset = []
for idx in range(len(X_train)):
    train_dataset.append((X_train[idx], Y_train[idx]))

X_test = mnist.X_test
Y_test = mnist.Y_test

# Create batches of training dataset
np.random.shuffle(train_dataset)
train_dataset = create_dataset(train_dataset, BATCH_SIZE)

learning_rate = tf.placeholder(tf.float32, shape=[])
lr = LEARNING_RATE_0

model_instance = Model()
model = model_instance.model

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

optimizer = tf.keras.optimizers.SGD(
    learning_rate=LEARNING_RATE_0
)

train_loss_results = []
train_accuracy_results = []

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(EPOCHS):
        for batch_idx, (X_train_batch, Y_train_batch) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(X_train_batch)
                loss_value = loss_fn(Y_train_batch, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if batch_idx % 25 == 0:
                loss_value = loss_value.eval()
                print('Batch: {}/{}\tLoss: {}'.format(batch_idx+1, len(train_dataset), float(loss_value)))

        print("Epoch {}".format(epoch))
