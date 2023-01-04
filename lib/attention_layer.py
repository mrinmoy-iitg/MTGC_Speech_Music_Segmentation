#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 23:59:55 2022

@author: Mrinmoy Bhattacharjee, Ph.D. Scholar, IIT Guwahati, Assam, India

@source: https://srome.github.io/Understanding-Attention-in-Neural-Networks-Mathematically/
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow import keras


class MrinSelfAttention(layers.Layer):
    def __init__(self, initializer='he_normal', attention_dim=1, **kwargs):
        super(MrinSelfAttention, self).__init__(**kwargs)
        self.initializer = keras.initializers.get(initializer)
        self.attention_dim = attention_dim

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.query_kernel = self.add_weight(
            shape=(int(input_shape[self.attention_dim]), int(input_shape[self.attention_dim])),
            name = 'query_kernel',
            initializer=self.initializer,
            trainable=True
            )

        self.query_bias = self.add_weight(
            shape=(int(input_shape[self.attention_dim]),), 
            name = 'query_bias',
            initializer=self.initializer,
            trainable=True
            )

        self.key_kernel = self.add_weight(
            shape=(int(input_shape[self.attention_dim]), 1),
            name = 'key_kernel',
            initializer=self.initializer,
            trainable=True
            )

    def call(self, x, reduce_sum=True):
        if self.attention_dim==1:
            x = K.permute_dimensions(x, [0,2,1]) # shape = (None, feat_dim, time_steps)
        u_t = tf.matmul(x, self.query_kernel) + self.query_bias
        u_t = tf.nn.tanh(u_t)
        alpha_t = tf.matmul(u_t, self.key_kernel)
        alpha_t = tf.nn.softmax(alpha_t)
        s_t = tf.multiply(x, alpha_t)
        if self.attention_dim==1:
            s_t = K.permute_dimensions(s_t, [0,2,1]) # shape = (None, feat_dim, time_steps)
        if reduce_sum:
            output = tf.reduce_sum(s_t, axis=self.attention_dim)
        else:
            output = s_t
        return output

    def get_config(self):
        base_config = super(MrinSelfAttention, self).get_config()
        config = {
            'attention_dim': self.attention_dim,
            'initializer': keras.initializers.serialize(self.initializer),
            }
        return dict(list(base_config.items()) + list(config.items()))

'''

"""
## Let's test-drive it on MNIST
"""
# Training parameters
batch_size = 128
num_classes = 10
epochs = 20

# The data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# x_train = x_train.reshape(-1, 784)
# x_test = x_test.reshape(-1, 784)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

print('\n\n\n\n\n\n\n\n')

inp_lyr = keras.Input(shape=(28,28,))
x = MrinAttention(attention_dim=1)(inputs=[inp_lyr, K.permute_dimensions(inp_lyr, [0,2,1])])
x = layers.Dense(256)(x)
x = layers.Dropout(0.5)(x)
opt_lyr = layers.Dense(10)(x)

model = keras.models.Model(inp_lyr, opt_lyr)

# Compile the model
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

print(model.summary())

# # Train the model
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.15)

# # Test the model
# model.evaluate(x_test, y_test)

'''