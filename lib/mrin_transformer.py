#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 17:11:46 2022

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati

Code was obtained from the following source:
    https://www.tensorflow.org/text/tutorials/transformer

"""

import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
from lib.attention_layer import MrinSelfAttention as attn_lyr
from tensorflow.keras.layers import Layer, Input, Flatten, BatchNormalization, Dropout, Concatenate, Add, Dense, LayerNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K



def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)



def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    
    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k). Defaults to None.
    
    Returns:
        output, attention_weights
    """
    
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    
    # scale matmul_qk
    dk = tf.cast(tf.keras.backend.int_shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
        
    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    
    return output, attention_weights



def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        Dense(d_model)  # (batch_size, seq_len, d_model)
        ])



class MultiHeadAttention(Layer):
    
    def __init__(self, embed_size, d_model, num_heads, name='multi_head_attention'):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        self.embed_size = embed_size
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        
        self.dense = Dense(embed_size)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        
        concat_attention = tf.reshape(scaled_attention,(batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, embed_size)
        
        return output, attention_weights


class EncoderLayer(Layer):
    def __init__(self,*, embed_size, d_model, num_heads, dff, rate=0.1, output_size=None):
        super(EncoderLayer, self).__init__()
        self.output_size = output_size
        
        self.mha = MultiHeadAttention(embed_size=embed_size, d_model=d_model, num_heads=num_heads)
        
        if not self.output_size:
            self.ffn = point_wise_feed_forward_network(embed_size, dff)
        else:
            self.ffn = point_wise_feed_forward_network(output_size, dff)
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        
    def call(self, v, k, q, mask):
        attn_output, _ = self.mha(v, k=k, q=q, mask=mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(v + attn_output)  # (batch_size, input_seq_len, d_model)
        
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        if not self.output_size:
            out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        else:
            out2 = self.layernorm2(ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2



class DecoderLayer(Layer):
    def __init__(self,*, embed_size, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        
        self.mha1 = MultiHeadAttention(embed_size=embed_size, d_model=d_model, num_heads=num_heads)
        self.mha2 = MultiHeadAttention(embed_size=embed_size, d_model=d_model, num_heads=num_heads)
        
        self.ffn = point_wise_feed_forward_network(embed_size, dff)
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)
        
    def call(self, v, k, q, enc_output, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(v, k=k, q=q, mask=look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + v)
        
        attn2, attn_weights_block2 = self.mha2(enc_output, k=enc_output, q=out1, mask=padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
        
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, embed_size)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, embed_size)
        
        return out3, attn_weights_block1, attn_weights_block2



class Encoder(Layer):
    def __init__(self,*, num_layers, embed_size, d_model, num_heads, dff, max_tokens, rate=0.1, output_size=None):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.pos_encoding = positional_encoding(max_tokens, embed_size)
        
        self.enc_layers = [EncoderLayer(embed_size=embed_size, d_model=d_model, num_heads=num_heads, dff=dff, rate=rate, output_size=output_size) for _ in range(num_layers)]
        self.dropout = Dropout(rate)
        
    def call(self, v, k, q, mask):
        seq_len = tf.shape(v)[1]
        
        # adding position encoding.
        v += self.pos_encoding[:, :seq_len, :]
        
        v = self.dropout(v)
        
        for i in range(self.num_layers):
            v = self.enc_layers[i](v, k=k, q=q, mask=mask)
            
        return v  # (batch_size, input_seq_len, d_model)




class Decoder(Layer):
    def __init__(self,*, num_layers, embed_size, d_model, num_heads, dff, max_tokens, rate=0.1):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.pos_encoding = positional_encoding(max_tokens, embed_size)
        
        self.dec_layers = [DecoderLayer(embed_size=embed_size, d_model=d_model, num_heads=num_heads, dff=dff, rate=rate) for _ in range(num_layers)]
        self.dropout = Dropout(rate)
        
    def call(self, enc_output, v, k, q, look_ahead_mask, padding_mask):
        seq_len = tf.shape(v)[1]
        attention_weights = {}
        
        v += self.pos_encoding[:, :seq_len, :]
        
        v = self.dropout(v)
        
        for i in range(self.num_layers):
            v, block1, block2 = self.dec_layers[i](
                v,
                k=k,
                q=q,
                enc_output=enc_output, 
                look_ahead_mask=look_ahead_mask, 
                padding_mask=padding_mask)
            
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2
            
            # x.shape == (batch_size, target_seq_len, d_model)
        
        return v, attention_weights



# This allows to the transformer to know where there is real data and where it is padded
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)



def create_look_ahead_mask(shape):
    mask = 1 - tf.linalg.band_part(tf.ones(shape), -1, 0)
    return mask  # (seq_len, seq_len)



class Transformer:
    attention_weights = None
    
    def __init__(self, *, seq_len, embed_size, num_layers, d_model, num_heads, dff, num_outputs, rate=0.1):
        super().__init__()

        self.time_steps = seq_len
        self.embed_size = embed_size

        self.encoder = Encoder(
            num_layers=num_layers, 
            embed_size=embed_size,
            d_model=d_model, 
            num_heads=num_heads, 
            dff=dff, 
            max_tokens=self.time_steps,
            rate=rate)
        
        self.decoder = Decoder(
            num_layers=num_layers, 
            embed_size=embed_size,
            d_model=d_model,
            num_heads=num_heads, 
            dff=dff,
            max_tokens=self.time_steps,
            rate=rate)
                
        self.final_layer = Dense(units=num_outputs, activation='sigmoid', name='genre_outputs')
        
        
    def get_model(self, input_name):
        # Keras models prefer if you pass all your inputs in the first argument
        value_inp = Input(shape=(self.time_steps, self.embed_size), name='cbow_feat_input')
        key_inp = Input(shape=(self.time_steps, 240), name='stats_feat_input')
        query_inp = Input(shape=(self.time_steps, 2), name='sm_pred_input')

        padding_mask, look_ahead_mask = self.create_masks(value_inp, value_inp)
        
        enc_output = self.encoder(
            value_inp,
            k=key_inp,
            q=query_inp,
            mask=None)  # (batch_size, inp_seq_len, d_model)
        
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, self.attention_weights = self.decoder(
            enc_output, 
            v=value_inp, 
            k=key_inp,
            q=query_inp,
            look_ahead_mask=None, 
            padding_mask=None)
        
        flattened_output = Flatten()(dec_output)
        
        final_output = self.final_layer(flattened_output)  # (batch_size, num_outputs)

        return Model(inputs={'cbow_feat_input':value_inp, 'stats_feat_input':key_inp, 'sm_pred_input':query_inp}, outputs=[final_output], name='genre_classifier')
    
    def create_masks(self, inp, tar):
        # Encoder padding mask (Used in the 2nd attention block in the decoder too.)
        padding_mask = create_padding_mask(inp)
        
        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask((tf.keras.backend.int_shape(tar)[1], tf.keras.backend.int_shape(tar)[2]))
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = look_ahead_mask[tf.newaxis, tf.newaxis, :]
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        
        return padding_mask, look_ahead_mask



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)






class Transformer_Fusion_hierarchical_attn:
    attention_weights = None
    
    def __init__(self, *, seq_len, embed_size, num_layers, d_model, num_heads, dff, num_outputs, rate=0.1):
        super().__init__()

        self.time_steps = seq_len
        self.embed_size = embed_size
        self.do_rate = rate
        
        self.encoder = {}
        self.encoder[0] = Encoder( # cbow, stats
                num_layers=num_layers, 
                embed_size=200,
                d_model=d_model, 
                num_heads=num_heads, 
                dff=dff, 
                max_tokens=self.time_steps,
                rate=self.do_rate)
        self.encoder[1] = Encoder( # cbow, smpred
                num_layers=num_layers, 
                embed_size=200,
                d_model=d_model, 
                num_heads=num_heads, 
                dff=dff, 
                max_tokens=self.time_steps,
                rate=self.do_rate)
        self.encoder[2] = Encoder( # stats, cbow
                num_layers=num_layers, 
                embed_size=240,
                d_model=d_model, 
                num_heads=num_heads, 
                dff=dff, 
                max_tokens=self.time_steps,
                rate=self.do_rate)
        self.encoder[3] = Encoder( # stats, smpred
                num_layers=num_layers, 
                embed_size=240,
                d_model=d_model, 
                num_heads=num_heads, 
                dff=dff, 
                max_tokens=self.time_steps,
                rate=self.do_rate)
        self.encoder[4] = Encoder( # smpred, cbow
                num_layers=num_layers, 
                embed_size=2,
                d_model=d_model, 
                num_heads=num_heads, 
                dff=dff, 
                max_tokens=self.time_steps,
                rate=self.do_rate)
        self.encoder[5] = Encoder( # smpred, stats
                num_layers=num_layers, 
                embed_size=2,
                d_model=d_model, 
                num_heads=num_heads, 
                dff=dff, 
                max_tokens=self.time_steps,
                rate=self.do_rate)
        
        self.encoder_lev2 = {}
        self.encoder_lev2[0] = Encoder( # cbow_stats_enc, cbow_smpred_enc 
                num_layers=num_layers, 
                embed_size=200,
                d_model=d_model, 
                num_heads=num_heads, 
                dff=dff, 
                max_tokens=self.time_steps,
                rate=self.do_rate)

        self.encoder_lev2[1] = Encoder( # stats_cbow_enc, stats_smpred_enc 
                num_layers=num_layers, 
                embed_size=240,
                d_model=d_model, 
                num_heads=num_heads, 
                dff=dff, 
                max_tokens=self.time_steps,
                rate=self.do_rate)

        self.encoder_lev2[2] = Encoder( # smpred_cbow_enc, smpred_stats_enc 
                num_layers=num_layers, 
                embed_size=2,
                d_model=d_model, 
                num_heads=num_heads, 
                dff=dff, 
                max_tokens=self.time_steps,
                rate=self.do_rate)


        self.encoder_lev3 = Encoder(
                num_layers=num_layers, 
                embed_size=440,
                d_model=d_model, 
                num_heads=num_heads, 
                dff=dff, 
                max_tokens=self.time_steps,
                rate=self.do_rate)

        self.comb_transform = Dense(units=embed_size, activation='relu')
                        
        self.final_layer = Dense(units=num_outputs, activation='sigmoid', name='genre_outputs')

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(self.do_rate)


        
    def get_model(self):
        cbow_inp = Input(shape=(self.time_steps, 200), name='cbow_feat_input')
        stats_inp = Input(shape=(self.time_steps, 240), name='stats_feat_input')
        smpred_inp = Input(shape=(self.time_steps, 2), name='sm_pred_input')
        
        ''' Level 1 '''
        enc_output_0 = self.encoder[0](
            cbow_inp,
            k=cbow_inp,
            q=stats_inp,
            mask=None)  # (batch_size, inp_seq_len, 200)
        enc_output_1 = self.encoder[1](
            cbow_inp,
            k=cbow_inp,
            q=smpred_inp,
            mask=None)  # (batch_size, inp_seq_len, 200)
        enc_output_2 = self.encoder[2](
            stats_inp,
            k=stats_inp,
            q=cbow_inp,
            mask=None)  # (batch_size, inp_seq_len, 240)
        enc_output_3 = self.encoder[3](
            stats_inp,
            k=stats_inp,
            q=smpred_inp,
            mask=None)  # (batch_size, inp_seq_len, 240)
        enc_output_4 = self.encoder[4](
            smpred_inp,
            k=smpred_inp,
            q=cbow_inp,
            mask=None)  # (batch_size, inp_seq_len, 2)
        enc_output_5 = self.encoder[5](
            smpred_inp,
            k=smpred_inp,
            q=stats_inp,
            mask=None)  # (batch_size, inp_seq_len, 2)
        
        concat_enc_output_lev1 = Concatenate(axis=-1)([enc_output_0, enc_output_1, enc_output_2, enc_output_3, enc_output_4, enc_output_5])  # (batch_size, inp_seq_len, 884)
        concat_enc_output_lev1 = BatchNormalization()(concat_enc_output_lev1)
        concat_enc_output_lev1 = Dropout(self.do_rate)(concat_enc_output_lev1)
        
        
        ''' Level 2 '''
        enc_lev2_output_0 = self.encoder_lev2[0](
            enc_output_0,
            k=enc_output_0,
            q=enc_output_1,
            mask=None)  # (batch_size, inp_seq_len, 200)
        enc_lev2_output_1 = self.encoder_lev2[1](
            enc_output_2,
            k=enc_output_2,
            q=enc_output_3,
            mask=None)  # (batch_size, inp_seq_len, 240)
        enc_lev2_output_2 = self.encoder_lev2[2](
            enc_output_4,
            k=enc_output_4,
            q=enc_output_5,
            mask=None)  # (batch_size, inp_seq_len, 2)

        concat_enc_output_lev2 = Concatenate(axis=-1)([enc_lev2_output_0, enc_lev2_output_1, enc_lev2_output_2, concat_enc_output_lev1])  # (batch_size, inp_seq_len, 1326)
        concat_enc_output_lev2 = BatchNormalization()(concat_enc_output_lev2)
        concat_enc_output_lev2 = Dropout(self.do_rate)(concat_enc_output_lev2)

        
        
        ''' Level 3 '''
        concat_enc_output_lev2_enc0_enc1 = Concatenate(axis=-1)([enc_lev2_output_0, enc_lev2_output_1])  # (batch_size, inp_seq_len, 440)
        enc_lev3_output = self.encoder_lev3(
            concat_enc_output_lev2_enc0_enc1,
            k=concat_enc_output_lev2_enc0_enc1,
            q=enc_lev2_output_2,
            mask=None)  # (batch_size, inp_seq_len, 440)
        
        concat_enc_output_lev3 = Concatenate(axis=-1)([enc_lev3_output, concat_enc_output_lev2])  # (batch_size, inp_seq_len, 1766)
        concat_enc_output_lev3 = BatchNormalization()(concat_enc_output_lev3)
        concat_enc_output_lev3 = Dropout(self.do_rate)(concat_enc_output_lev3)


        
        ''' Level 4 '''
        mha_output, _ = MultiHeadAttention(embed_size=1766, d_model=128, num_heads=16)(
            concat_enc_output_lev3, 
            k=concat_enc_output_lev3, 
            q=concat_enc_output_lev3, 
            mask=None)  # (batch_size, input_seq_len, 1766)
        mha_output = Dropout(0.1)(mha_output)  # (batch_size, inp_seq_len, 1766)
        mha_output = self.layernorm1(concat_enc_output_lev3 + mha_output)  # (batch_size, input_seq_len, 1766)
        
        transformed_output_lev4 = self.comb_transform(mha_output)  # (batch_size, inp_seq_len, embed_size)
        transformed_output_lev4 = BatchNormalization()(transformed_output_lev4)  # (batch_size, inp_seq_len, embed_size)
        transformed_output_lev4 = Dropout(self.do_rate)(transformed_output_lev4)  # (batch_size, inp_seq_len, embed_size)
        
        
        # flattened_output = Flatten()(transformed_output)  # (batch_size, inp_seq_len*embed_size)

        x_lev1_time = attn_lyr(attention_dim=1)(concat_enc_output_lev1, reduce_sum=True) # (None, 884)
        x_lev1_time = BatchNormalization(axis=-1)(x_lev1_time)
        x_lev1_feat = attn_lyr(attention_dim=2)(concat_enc_output_lev1, reduce_sum=True) # (None, 30)
        x_lev1_feat = BatchNormalization(axis=-1)(x_lev1_feat)        
        x_lev1 = Concatenate(axis=-1)([x_lev1_time, x_lev1_feat]) # (None, 914)

        x_lev2_time = attn_lyr(attention_dim=1)(concat_enc_output_lev2, reduce_sum=True) # (None, 1326)
        x_lev2_time = BatchNormalization(axis=-1)(x_lev2_time)
        x_lev2_feat = attn_lyr(attention_dim=2)(concat_enc_output_lev2, reduce_sum=True) # (None, 30)
        x_lev2_feat = BatchNormalization(axis=-1)(x_lev2_feat)        
        x_lev2 = Concatenate(axis=-1)([x_lev2_time, x_lev2_feat]) # (None, 1356)
        
        x_lev3_time = attn_lyr(attention_dim=1)(concat_enc_output_lev3, reduce_sum=True) # (None, 1766)
        x_lev3_time = BatchNormalization(axis=-1)(x_lev3_time)
        x_lev3_feat = attn_lyr(attention_dim=2)(concat_enc_output_lev3, reduce_sum=True) # (None, 30)
        x_lev3_feat = BatchNormalization(axis=-1)(x_lev3_feat)        
        x_lev3 = Concatenate(axis=-1)([x_lev3_time, x_lev3_feat]) # (None, 1796)

        x_lev4_time = attn_lyr(attention_dim=1)(transformed_output_lev4, reduce_sum=True) # (None, 200)
        x_lev4_time = BatchNormalization(axis=-1)(x_lev4_time)
        x_lev4_feat = attn_lyr(attention_dim=2)(transformed_output_lev4, reduce_sum=True) # (None, 30)
        x_lev4_feat = BatchNormalization(axis=-1)(x_lev4_feat)
        x_lev4 = Concatenate(axis=-1)([x_lev4_time, x_lev4_feat]) # (None, 230)

        x_all_lev = Concatenate(axis=-1)([x_lev1, x_lev2, x_lev3, x_lev4]) # (None, 4296) # (None, 4206)
        x_all_lev = BatchNormalization(axis=-1)(x_all_lev)        
        final_output = self.final_layer(x_all_lev)  # (batch_size, num_outputs)

        return Model(
            inputs={
                'cbow_feat_input':cbow_inp, 
                'stats_feat_input':stats_inp, 
                'sm_pred_input':smpred_inp}, 
            outputs=[final_output], 
            name='genre_classifier'
            )





class Transformer_Encoder:
    def __init__(self, *, seq_len, embed_size, num_layers, d_model, num_heads, dff, num_outputs, rate=0.1):
        super().__init__()

        self.time_steps = seq_len
        self.embed_size = embed_size

        self.encoder0 = Encoder(
            num_layers=num_layers, 
            embed_size=200,
            d_model=d_model, 
            num_heads=num_heads, 
            dff=dff, 
            max_tokens=self.time_steps,
            rate=rate)

        self.encoder1 = Encoder(
            num_layers=num_layers, 
            embed_size=240,
            d_model=d_model, 
            num_heads=num_heads, 
            dff=dff, 
            max_tokens=self.time_steps,
            rate=rate)

        self.encoder2 = Encoder(
            num_layers=num_layers, 
            embed_size=2,
            d_model=d_model, 
            num_heads=num_heads, 
            dff=dff, 
            max_tokens=self.time_steps,
            rate=rate)
                        
        self.final_layer = Dense(units=num_outputs, activation='sigmoid', name='genre_outputs')
        
        
    def get_model(self, input_name):
        # Keras models prefer if you pass all your inputs in the first argument
        cbow_inp = Input(shape=(self.time_steps, 200), name='cbow_feat_input')
        stats_inp = Input(shape=(self.time_steps, 240), name='stats_feat_input')
        smpred_inp = Input(shape=(self.time_steps, 2), name='sm_pred_input')
        
        enc_output0 = self.encoder0(
            cbow_inp,
            k=cbow_inp,
            q=cbow_inp,
            mask=None) # (batch_size, inp_seq_len, 200)
        
        enc_output1 = self.encoder1(
            stats_inp,
            k=stats_inp,
            q=stats_inp,
            mask=None) # (batch_size, inp_seq_len, 240)

        enc_output2 = self.encoder2(
            smpred_inp,
            k=smpred_inp,
            q=smpred_inp,
            mask=None) # (batch_size, inp_seq_len, 2)
        
        x_enc = Concatenate(axis=-1)([enc_output0, enc_output1, enc_output2]) # (batch_size, inp_seq_len, 442)

        x_attn = attn_lyr(attention_dim=1)(x_enc, reduce_sum=True) # (None, 442)
        x_attn = BatchNormalization(axis=-1)(x_attn)
                        
        final_output = self.final_layer(x_attn)  # (batch_size, num_outputs)

        return Model(inputs={
            'cbow_feat_input':cbow_inp, 
            'stats_feat_input':stats_inp, 
            'sm_pred_input':smpred_inp}, 
            outputs=[final_output], 
            name='genre_classifier'
            )

    
    def create_masks(self, inp, tar):
        # Encoder padding mask (Used in the 2nd attention block in the decoder too.)
        padding_mask = create_padding_mask(inp)
        
        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask((tf.keras.backend.int_shape(tar)[1], tf.keras.backend.int_shape(tar)[2]))
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = look_ahead_mask[tf.newaxis, tf.newaxis, :]
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        
        return padding_mask, look_ahead_mask




class Transformer_Encoder_var_input:
    def __init__(self, *, seq_len, embed_size, num_layers, d_model, num_heads, dff, num_outputs, feat_names, rate=0.1):
        super().__init__()

        self.time_steps = seq_len
        self.embed_size = embed_size
        self.feat_names = feat_names
        
        self.encoder = {}
        for i in range(len(self.embed_size)):
            self.encoder[i] = Encoder(
                num_layers=num_layers, 
                embed_size=self.embed_size[i],
                d_model=d_model, 
                num_heads=num_heads, 
                dff=dff, 
                max_tokens=self.time_steps,
                rate=rate)
                        
        self.final_layer = Dense(units=num_outputs, activation='sigmoid', name='genre_outputs')
        
        
    def get_model(self, input_name):
        # Keras models prefer if you pass all your inputs in the first argument
        
        inp_lyrs = {}
        enc_output = {}
        for i in range(len(self.embed_size)):
            inp_lyrs[self.feat_names[i]] = Input(shape=(self.time_steps, self.embed_size[i]), name=self.feat_names[i])
            enc_output[i] = self.encoder[i](
                inp_lyrs[self.feat_names[i]],
                k=inp_lyrs[self.feat_names[i]],
                q=inp_lyrs[self.feat_names[i]],
                mask=None) # (batch_size, inp_seq_len, 512)
            if i==0:
                concat_enc_output = enc_output[i]
            else:
                concat_enc_output = Concatenate(axis=-1)([concat_enc_output, enc_output[i]])
        
        x_attn = attn_lyr(attention_dim=1)(concat_enc_output, reduce_sum=True) # (None, 512)
        x_attn = BatchNormalization(axis=-1)(x_attn)
                        
        final_output = self.final_layer(x_attn)  # (batch_size, num_outputs)
        
        return Model(
            inputs=inp_lyrs,
            outputs=[final_output], 
            name='genre_classifier'
            )




class Transformer_Encoder_Early_Fusion:
    def __init__(self, *, seq_len, embed_size, num_layers, d_model, num_heads, dff, num_outputs, rate=0.1):
        super().__init__()

        self.time_steps = seq_len
        self.embed_size = embed_size

        self.encoder = Encoder(
            num_layers=num_layers, 
            embed_size=442,
            d_model=d_model, 
            num_heads=num_heads, 
            dff=dff, 
            max_tokens=self.time_steps,
            rate=rate,
            output_size=None,
            )

        self.final_layer = Dense(units=num_outputs, activation='sigmoid', name='genre_outputs')
        
        
    def get_model(self, input_name):
        # Keras models prefer if you pass all your inputs in the first argument
        cbow_inp = Input(shape=(self.time_steps, 200), name='cbow_feat_input')
        stats_inp = Input(shape=(self.time_steps, 240), name='stats_feat_input')
        smpred_inp = Input(shape=(self.time_steps, 2), name='sm_pred_input')
        
        early_fusion = Concatenate(axis=-1)([cbow_inp, stats_inp, smpred_inp])
        
        enc_output = self.encoder(
            early_fusion,
            k=early_fusion,
            q=early_fusion,
            mask=None) # (batch_size, inp_seq_len, 442)
        
        flattened_enc_output = Flatten()(enc_output)
                                
        final_output = self.final_layer(flattened_enc_output)  # (batch_size, num_outputs)

        return Model(inputs={
            'cbow_feat_input':cbow_inp, 
            'stats_feat_input':stats_inp, 
            'sm_pred_input':smpred_inp}, 
            outputs=[final_output], 
            name='genre_classifier'
            )
