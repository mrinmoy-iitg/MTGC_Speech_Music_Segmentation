#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 23:59:55 2022

@author: Mrinmoy Bhattacharjee, Ph.D. Scholar, IIT Guwahati, Assam, India

@reference: https://srome.github.io/Understanding-Attention-in-Neural-Networks-Mathematically/
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
