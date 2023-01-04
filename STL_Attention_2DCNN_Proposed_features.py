#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 13:20:59 2022

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
np.random.seed(1989)
import os
import datetime
import lib.misc as misc
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, BatchNormalization, Input, Dropout, Activation, Conv2D, MaxPooling2D, Concatenate, Cropping2D, Conv1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
import time
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import average_precision_score as APS
import sys
from lib.attention_layer import MrinSelfAttention as attn_lyr
from sklearn.preprocessing import StandardScaler
from scipy.signal import medfilt






def start_GPU_session():
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.4)
    config = tf.compat.v1.ConfigProto(
        device_count={'GPU': 1 , 'CPU': 1}, 
        gpu_options=gpu_options,
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        )
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    tf.compat.v1.random.set_random_seed(1989)



def reset_TF_session():
    tf.compat.v1.keras.backend.clear_session()




def cnn_model(PARAMS):
    nConv_cbow = 80
    nConv_stats = 80
    nConv_raw = 50
    act = 'relu'
    do = 0.1
    n_fc_nodes = 300
    n_fc_lyrs = 3
    num_output = len(PARAMS['genre_list'])
    learning_rate = 0.001

    if PARAMS['raw_feat_input']:
        raw_feat_input = Input((128,1407,), name='raw_feat_input')
        x_raw = K.expand_dims(raw_feat_input, axis=3) # (None, 128, 1407, 1)
    
        x_raw = Conv2D(nConv_raw, kernel_size=(3,3), strides=(1,1), padding='same')(x_raw) # (None, 128, 1407, 1)
        x_raw = BatchNormalization(axis=-1)(x_raw)
        
        x_raw = MaxPooling2D(pool_size=(3,4), strides=(3,4), padding='valid')(x_raw) # (None, 42, 351, nConv)
    
        x_raw = Conv2D(nConv_raw, kernel_size=(3,3), strides=(1,1), padding='same', dilation_rate=(1,2))(x_raw) # (None, 42, 351, nConv)
        x_raw = BatchNormalization(axis=-1)(x_raw)

        x_raw = MaxPooling2D(pool_size=(3,4), strides=(3,4), padding='same')(x_raw) # (None, 14, 88, 2*nConv)
    
        x_raw = Conv2D(nConv_raw, kernel_size=(3,3), strides=(1,1), padding='same', dilation_rate=(1,4))(x_raw) # (None, 14, 88, 1.5*nConv)
        x_raw = BatchNormalization(axis=-1)(x_raw)
    
        x_raw = MaxPooling2D(pool_size=(3,3), strides=(3,3), padding='same')(x_raw) # (None, 5, 30, 3*nConv)
    
        x_raw = Conv2D(nConv_raw, kernel_size=(K.int_shape(x_raw)[1],3), strides=(K.int_shape(x_raw)[1],1), padding='same')(x_raw) # (None, 1, 30, 4*nConv)
        x_raw = BatchNormalization(axis=-1)(x_raw)

        x_raw = K.squeeze(x_raw, axis=1) # (None, 30, 4*nConv)

        x_raw_time = attn_lyr(attention_dim=1)(x_raw, reduce_sum=True) # (None, 4*nConv)
        x_raw_time = BatchNormalization(axis=-1)(x_raw_time)
        x_raw_feat = attn_lyr(attention_dim=2)(x_raw, reduce_sum=True) # (None, 4*nConv)
        x_raw_feat = BatchNormalization(axis=-1)(x_raw_feat)        
        x_raw = Concatenate(axis=-1)([x_raw_time, x_raw_feat])
        
        x = x_raw

            
    if PARAMS['cbow_feat_input']:
        seg_size = int(PARAMS['W']/1000)
        cbow_input = Input(shape=(PARAMS['feat_dim'],seg_size,1), name='cbow_feat_input') # (None, 200, 30, 1)
        
        ''' CBoW-ASPT '''
        x_cbow_aspt = Cropping2D(cropping=((0, int(PARAMS['feat_dim']/2)), (0, 0)))(cbow_input) # (None, 100, 30, 1)

        x_cbow_aspt = Conv2D(nConv_cbow, kernel_size=(3,3), strides=(1,1), padding='same')(x_cbow_aspt) # (None, 100, 30, 10)
        x_cbow_aspt = BatchNormalization(axis=-1)(x_cbow_aspt)
        
        x_cbow_aspt = MaxPooling2D(pool_size=(3,1), strides=(3,1), padding='valid')(x_cbow_aspt) # (None, 33, 30, 10)
    
        x_cbow_aspt = Conv2D(nConv_cbow, kernel_size=(3,3), strides=(1,1), padding='same' , dilation_rate=(1,2))(x_cbow_aspt) # (None, 33, 30, 20)
        x_cbow_aspt = BatchNormalization(axis=-1)(x_cbow_aspt)
    
        x_cbow_aspt = MaxPooling2D(pool_size=(3,1), strides=(3,1), padding='valid')(x_cbow_aspt) # (None, 11, 30, 10)
    
        x_cbow_aspt = Conv2D(nConv_cbow, kernel_size=(3,3), strides=(1,1), padding='same' , dilation_rate=(1,4))(x_cbow_aspt) # (None, 11, 30, 30)
        x_cbow_aspt = BatchNormalization(axis=-1)(x_cbow_aspt)
    
        x_cbow_aspt = MaxPooling2D(pool_size=(3,1), strides=(3,1), padding='valid')(x_cbow_aspt) # (None, 3, 30, 30)
        
        x_cbow_aspt = Conv2D(nConv_cbow, kernel_size=(K.int_shape(x_cbow_aspt)[1],3), strides=(K.int_shape(x_cbow_aspt)[1],1), padding='same')(x_cbow_aspt) # (None, 1, 30, 40)
        x_cbow_aspt = BatchNormalization(axis=-1)(x_cbow_aspt)

        x_cbow_aspt = K.squeeze(x_cbow_aspt, axis=1) # (None, 30, 50)

        
        ''' CBoW-LSPT '''
        x_cbow_lspt = Cropping2D(cropping=((int(PARAMS['feat_dim']/2), 0), (0, 0)))(cbow_input) # (None, 100, 30, 1)

        x_cbow_lspt = Conv2D(nConv_cbow, kernel_size=(3,3), strides=(1,1), padding='same')(x_cbow_lspt) # (None, 100, 30, 10)
        x_cbow_lspt = BatchNormalization(axis=-1)(x_cbow_lspt)
        
        x_cbow_lspt = MaxPooling2D(pool_size=(3,1), strides=(3,1), padding='valid')(x_cbow_lspt) # (None, 33, 30, 10)
    
        x_cbow_lspt = Conv2D(nConv_cbow, kernel_size=(3,3), strides=(1,1), padding='same' , dilation_rate=(1,2))(x_cbow_lspt) # (None, 33, 30, 20)
        x_cbow_lspt = BatchNormalization(axis=-1)(x_cbow_lspt)

        x_cbow_lspt = MaxPooling2D(pool_size=(3,1), strides=(3,1), padding='valid')(x_cbow_lspt) # (None, 11, 30, 10)
    
        x_cbow_lspt = Conv2D(nConv_cbow, kernel_size=(3,3), strides=(1,1), padding='same' , dilation_rate=(1,4))(x_cbow_lspt) # (None, 11, 30, 30)
        x_cbow_lspt = BatchNormalization(axis=-1)(x_cbow_lspt)
    
        x_cbow_lspt = MaxPooling2D(pool_size=(3,1), strides=(3,1), padding='valid')(x_cbow_lspt) # (None, 3, 30, 30)
        
        x_cbow_lspt = Conv2D(nConv_cbow, kernel_size=(K.int_shape(x_cbow_lspt)[1],3), strides=(K.int_shape(x_cbow_lspt)[1],1), padding='same')(x_cbow_lspt) # (None, 1, 30, 40)
        x_cbow_lspt = BatchNormalization(axis=-1)(x_cbow_lspt)

        x_cbow_lspt = K.squeeze(x_cbow_lspt, axis=1) # (None, 30, 50)

        x_cbow_merged = Concatenate(axis=-1)([x_cbow_aspt, x_cbow_lspt])
        x_cbow_time = attn_lyr(attention_dim=1)(x_cbow_merged, reduce_sum=True) # (None, 30, 4*nConv)
        x_cbow_time = BatchNormalization(axis=-1)(x_cbow_time)
        x_cbow_feat = attn_lyr(attention_dim=2)(x_cbow_merged, reduce_sum=True) # (None, 30, 4*nConv)
        x_cbow_feat = BatchNormalization(axis=-1)(x_cbow_feat)
        x_cbow = Concatenate(axis=-1)([x_cbow_time, x_cbow_feat]) # (None, 4*nConv+30)
        # x_cbow = x_cbow_merged

        # x_cbow = K.expand_dims(x_cbow, axis=2) # (None, nConv_cbow+30,1)
        # x_cbow = Conv1D(1, kernel_size=1, strides=1, padding='same')(x_cbow)
        # x_cbow = K.squeeze(x_cbow, axis=2) # (None, nConv_cbow+30)

        try:
            x = Concatenate(axis=-1)([x, x_cbow]) # (None, 4*nConv+30)
        except:
            x = x_cbow


    if PARAMS['stats_feat_input']:
        seg_size = int(PARAMS['W']/1000)
        stats_feat_input = Input(shape=(240,seg_size,1), name='stats_feat_input')

        ''' stats-ASPT '''
        x_stats_aspt = Cropping2D(cropping=((0, 120), (0, 0)))(stats_feat_input) # (None, 120, 30, 1)
        
        x_stats_aspt = Conv2D(nConv_stats, kernel_size=(3,3), strides=(1,1), padding='same', dilation_rate=(1,2))(x_stats_aspt) # (None, 120, 30, nConv)
        x_stats_aspt = BatchNormalization(axis=-1)(x_stats_aspt)
        
        x_stats_aspt = MaxPooling2D(pool_size=(3,1), strides=(3,1), padding='valid')(x_stats_aspt) # (None, 40, 30, nConv)
    
        x_stats_aspt = Conv2D(nConv_stats, kernel_size=(3,3), strides=(1,1), padding='same', dilation_rate=(1,4))(x_stats_aspt) # (None, 40, 30, nConv)
        x_stats_aspt = BatchNormalization(axis=-1)(x_stats_aspt)
    
        x_stats_aspt = MaxPooling2D(pool_size=(3,1), strides=(3,1), padding='valid')(x_stats_aspt) # (None, 13, 30, 2*nConv)
        
        x_stats_aspt = Conv2D(nConv_stats, kernel_size=(3,3), strides=(1,1), padding='same')(x_stats_aspt) # (None, 13, 30, 3*nConv)
        x_stats_aspt = BatchNormalization(axis=-1)(x_stats_aspt)

        x_stats_aspt = MaxPooling2D(pool_size=(3,1), strides=(3,1), padding='valid')(x_stats_aspt) # (None, 4, 30, 2*nConv)

        x_stats_aspt = Conv2D(nConv_stats, kernel_size=(4,4), strides=(4,1), padding='same')(x_stats_aspt) # (None, 1, 30, 3*nConv)
        x_stats_aspt = BatchNormalization(axis=-1)(x_stats_aspt)

        x_stats_aspt = K.squeeze(x_stats_aspt, axis=1) # (None, 30, 4*nConv)

        
        ''' stats-LSPT '''
        x_stats_lspt = Cropping2D(cropping=((120, 0), (0, 0)))(stats_feat_input) # (None, 120, 30, 1)
        
        x_stats_lspt = Conv2D(nConv_stats, kernel_size=(3,3), strides=(1,1), padding='same', dilation_rate=(1,2))(x_stats_lspt) # (None, 120, 30, nConv)
        x_stats_lspt = BatchNormalization(axis=-1)(x_stats_lspt)
        
        x_stats_lspt = MaxPooling2D(pool_size=(3,1), strides=(3,1), padding='valid')(x_stats_lspt) # (None, 40, 30, nConv)
    
        x_stats_lspt = Conv2D(nConv_stats, kernel_size=(3,3), strides=(1,1), padding='same', dilation_rate=(1,4))(x_stats_lspt) # (None, 40, 30, nConv)
        x_stats_lspt = BatchNormalization(axis=-1)(x_stats_lspt)
    
        x_stats_lspt = MaxPooling2D(pool_size=(3,1), strides=(3,1), padding='valid')(x_stats_lspt) # (None, 13, 30, 2*nConv)
        
        x_stats_lspt = Conv2D(nConv_stats, kernel_size=(3,3), strides=(1,1), padding='same')(x_stats_lspt) # (None, 13, 30, 3*nConv)
        x_stats_lspt = BatchNormalization(axis=-1)(x_stats_lspt)

        x_stats_lspt = MaxPooling2D(pool_size=(3,1), strides=(3,1), padding='valid')(x_stats_lspt) # (None, 4, 30, 2*nConv)
        
        x_stats_lspt = Conv2D(nConv_stats, kernel_size=(4,4), strides=(4,1), padding='same')(x_stats_lspt) # (None, 1, 30, 3*nConv)
        x_stats_lspt = BatchNormalization(axis=-1)(x_stats_lspt)

        x_stats_lspt = K.squeeze(x_stats_lspt, axis=1) # (None, 30, 4*nConv)

        x_stats_merged = Concatenate(axis=-1)([x_stats_aspt, x_stats_lspt])
        x_stats_time = attn_lyr(attention_dim=1)(x_stats_merged, reduce_sum=True) # (None, 30, 4*nConv)
        x_stats_time = BatchNormalization(axis=-1)(x_stats_time)        
        x_stats_feat = attn_lyr(attention_dim=2)(x_stats_merged, reduce_sum=True) # (None, 30, 4*nConv)
        x_stats_feat = BatchNormalization(axis=-1)(x_stats_feat)
        x_stats = Concatenate(axis=-1)([x_stats_time, x_stats_feat])
        # x_stats = x_stats_merged

        # x_stats = K.expand_dims(x_stats, axis=2) # (None, nConv_stats+30,1)
        # x_stats = Conv1D(1, kernel_size=1, strides=1, padding='same')(x_stats)
        # x_stats = K.squeeze(x_stats, axis=2) # (None, nConv_stats+30)

        try:
            x = Concatenate(axis=-1)([x, x_stats])
        except:
            x = x_stats


    if PARAMS['sm_pred_input']:
        seg_size = int(PARAMS['W']/1000)
        sm_pred_input = Input((2,seg_size,), name='sm_pred_input')
        x_sm = K.permute_dimensions(sm_pred_input, [0,2,1])  # (None, 30, 2)
        
        x_sm_time = attn_lyr(attention_dim=1)(x_sm, reduce_sum=True) # (None, 30, 2)
        x_sm_time = BatchNormalization(axis=-1)(x_sm_time)
        x_sm_feat = attn_lyr(attention_dim=2)(x_sm, reduce_sum=True) # (None, 30, 2)
        x_sm_feat = BatchNormalization(axis=-1)(x_sm_feat)
        x_sm = Concatenate(axis=-1)([x_sm_time, x_sm_feat])
        # x_sm = x_sm_feat

        # x_sm = K.expand_dims(x_sm, axis=2) # (None, 2+30,1)
        # x_sm = Conv1D(1, kernel_size=1, strides=1, padding='same')(x_sm)
        # x_sm = K.squeeze(x_sm, axis=2) # (None, 2+30)
        
        try:
            x = Concatenate(axis=-1)([x, x_sm])
        except:
            x = x_sm
            
    # x = Flatten()(x)


    ''' Fully Connected layers '''
    x = Dense(n_fc_nodes)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation(act)(x)
    x = Dropout(do)(x)

    for n_fc in range(1, n_fc_lyrs):
        x = Dense(n_fc_nodes)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation(act)(x)
        x = Dropout(do)(x)

    inputs = {}
    if PARAMS['raw_feat_input']:
        inputs['raw_feat_input'] = raw_feat_input
    if PARAMS['cbow_feat_input']:
        inputs['cbow_feat_input'] = cbow_input
    if PARAMS['sm_pred_input']:
        inputs['sm_pred_input'] = sm_pred_input
    if PARAMS['stats_feat_input']:
        inputs['stats_feat_input'] = stats_feat_input

    output_layer = Dense(num_output, activation='sigmoid')(x)        
    model = Model(inputs, output_layer)
    opt = optimizers.Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[AUC(curve='PR')])
    
    return model, learning_rate





def get_raw_features(PARAMS, fName, nSeg):
    fv = misc.load_obj(PARAMS['raw_feature_opDir'], fName.split('.')[0])
    fv_reshaped = np.empty([])
    for seg_i in range(np.shape(fv)[0]):
        if np.size(fv_reshaped)<=1:
            fv_reshaped = fv[seg_i, :]
        else:
            fv_reshaped = np.append(fv_reshaped, fv[seg_i,:], axis=1)
    # print('fv_reshaped: ', np.shape(fv_reshaped), np.shape(fv))
    fv_reshaped = fv_reshaped.T
    fv_reshaped = StandardScaler().fit_transform(fv_reshaped)

    segFrames = 1407
    nFrames = np.shape(fv_reshaped)[0]
    fv_seg = np.empty([])
    frmStart = 0
    frmEnd = 0
    for frmStart in range(0, nFrames, int(nFrames/nSeg)):
        frmEnd = np.min([frmStart+segFrames, nFrames])
        if (frmEnd-frmStart)<segFrames:
            frmStart = frmEnd-segFrames
        if np.size(fv_seg)<=1:
            fv_seg = np.expand_dims(fv_reshaped[frmStart:frmEnd, :].T, axis=0)
        else:
            fv_seg = np.append(fv_seg, np.expand_dims(fv_reshaped[frmStart:frmEnd, :].T, axis=0), axis=0)
        frmStart = frmEnd
    fv_seg = fv_seg[:nSeg,:]
    
    return fv_seg





def reshape_CBoW_features(PARAMS, data, label, sm_pred, stats):
    reshaped_data = np.empty([])
    reshaped_label = np.empty([])
    reshaped_sm_pred = np.empty([])
    reshaped_stats = np.empty([])

    size = int(PARAMS['W']/1000) # 30s or 30000ms
    shift = int(PARAMS['W_shift']/1000) # 30s or 30000ms
    
    # print(f'{np.shape(data)}, {np.shape(sm_pred)}, {np.shape(msd)}, {np.shape(stats)}')
    if np.shape(data)[0]<size:
        data_orig = data.copy()
        label_orig = label.copy()
        sm_pred_orig = sm_pred.copy()
        stats_orig = stats.copy()
        while np.shape(data)[0]<size:
            data = np.append(data, data_orig, axis=0)
            label = np.append(label, label_orig, axis=0)
            sm_pred = np.append(sm_pred, sm_pred_orig, axis=0)
            stats = np.append(stats, stats_orig, axis=0)
    
    patch_labels = []
    for data_i in range(0, np.shape(data)[0], shift):
        data_j = np.min([np.shape(data)[0], data_i+size])
        if data_j-data_i<size:
            data_i = data_j-size
        if np.size(reshaped_data)<=1:
            reshaped_data = np.expand_dims(data[data_i:data_j,:].T, axis=0)
            reshaped_label = np.array(np.mean(label[data_i:data_j,:], axis=0).astype(int), ndmin=2)
            reshaped_sm_pred = np.expand_dims(sm_pred[data_i:data_j,:].T, axis=0)
            reshaped_stats = np.expand_dims(stats[data_i:data_j,:].T, axis=0)
        else:
            # print(f'sm_pred: {np.shape(reshaped_sm_pred)}, {np.shape(np.expand_dims(sm_pred[data_i:data_j,:].T, axis=0))}')
            reshaped_data = np.append(reshaped_data, np.expand_dims(data[data_i:data_j,:].T, axis=0), axis=0)
            reshaped_label = np.append(reshaped_label, np.array(np.mean(label[data_i:data_j,:], axis=0).astype(int), ndmin=2), axis=0)
            reshaped_sm_pred = np.append(reshaped_sm_pred, np.expand_dims(sm_pred[data_i:data_j,:].T, axis=0), axis=0)
            reshaped_stats = np.append(reshaped_stats, np.expand_dims(stats[data_i:data_j,:].T, axis=0), axis=0)

        patch_labels.append(np.mean(sm_pred[data_i:data_j,1]))
        # print('reshape_CBoW_features: ', data_i, np.shape(data), np.shape(reshaped_data), np.shape(reshaped_label), np.shape(reshaped_sm_pred), end='\r', flush=True)
        
    patch_labels = np.array(patch_labels)
    reshaped_data = np.expand_dims(reshaped_data, axis=3)
    reshaped_stats = np.expand_dims(reshaped_stats, axis=3)
    
    return reshaped_data, reshaped_label, reshaped_sm_pred, reshaped_stats, patch_labels




def generator(PARAMS, file_list, batchSize, add_noise=False):
    batch_count = 0

    file_list_temp = file_list[PARAMS['classes'][0]].copy()
    np.random.shuffle(file_list_temp)

    batchData_temp = np.empty([], dtype=float)
    batchData_stats_temp = np.empty([], dtype=float)
    sm_pred_temp = np.empty([], dtype=float)
    raw_feat_temp = np.empty([], dtype=float)
    batchLabel_temp = np.empty([], dtype=float)

    balance = 0
                
    while 1:    
        while balance<batchSize:
            if not file_list_temp:
                file_list_temp = file_list[PARAMS['classes'][0]].copy()
                np.random.shuffle(file_list_temp)
            fName = file_list_temp.pop()
            if not fName.split('.')[0] in PARAMS['annotations'].keys():
                # print(fName, ' not in annotations')
                continue
            if not os.path.exists(PARAMS['audio_path']+'/'+fName.split('/')[-1]):
                # print(fName, ' not in audio path')
                continue

            lab = np.zeros(len(PARAMS['genre_list']))
            for genre_i in PARAMS['genre_list'].keys():
                if genre_i in PARAMS['annotations'][fName.split('.')[0]]['genre']:
                    lab[PARAMS['genre_list'][genre_i]] = 1

            # To account for trailers with no labels
            if np.sum(lab)==0:
                # Augmenting minority genres
                # for genre_i in PARAMS['genre_list'].keys():
                #     if genre_i in PARAMS['minority_genres']:
                #         lab[PARAMS['genre_list'][genre_i]] = 1
                continue

            feat_fName = PARAMS['feature_path'] + '/' + fName.split('.')[0] + '.npy'
        
            fv = np.load(feat_fName)
            # fv = StandardScaler().fit_transform(fv)

            sm_pred_fName = PARAMS['sp_mu_pred_path'] + '/' + fName.split('.')[0] + '.npy'
            pred_sm = np.load(sm_pred_fName)
            if np.shape(pred_sm)[0]<np.shape(fv)[0]:
                pred_sm_orig = pred_sm.copy()
                while np.shape(pred_sm)[0]<np.shape(fv)[0]:
                    pred_sm = np.append(pred_sm, pred_sm_orig, axis=0)
                pred_sm = pred_sm[:np.shape(fv)[0], :]
            
            ''' Smoothing SM Predictions '''
            pred_sm_smooth = np.zeros(np.shape(pred_sm))
            pred_sm_smooth[:,1] = medfilt(pred_sm[:,1], kernel_size=PARAMS['smoothing_win_size'])
            pred_sm_smooth[:,0] = 1 - pred_sm_smooth[:,1]
            pred_sm = pred_sm_smooth.copy()

            stats_feat_fName = PARAMS['feature_path_Stats'] + '/' + fName.split('.')[0] + '.npy'
            fv_stats = np.load(stats_feat_fName)
            fv_stats = StandardScaler().fit_transform(fv_stats)

            lab = np.repeat(np.array(lab, ndmin=2), np.shape(fv)[0], axis=0)

            # print(f'{fName}, fv: {np.shape(fv)}, lab: {np.shape(lab)},  pred_sm: {np.shape(pred_sm)}')
            fv, lab, pred_sm, fv_stats, patch_labels = reshape_CBoW_features(PARAMS, fv, lab, pred_sm, fv_stats)
            # print('fv, lab, pred_sm: ', np.shape(fv), np.shape(lab), np.shape(pred_sm))

            if PARAMS['raw_feat_input']:
                fv_raw = get_raw_features(PARAMS, fName, np.shape(fv)[0])
                                                    
            if np.size(batchData_temp)<=1:
                batchData_temp = fv
                batchData_stats_temp = fv_stats
                batchLabel_temp = lab
                sm_pred_temp = pred_sm
                if PARAMS['raw_feat_input']:
                    raw_feat_temp = fv_raw
            else:
                batchData_temp = np.append(batchData_temp, fv, axis=0)
                batchData_stats_temp = np.append(batchData_stats_temp, fv_stats, axis=0)
                batchLabel_temp = np.append(batchLabel_temp, lab, axis=0)
                sm_pred_temp = np.append(sm_pred_temp, pred_sm, axis=0)
                if PARAMS['raw_feat_input']:
                    raw_feat_temp = np.append(raw_feat_temp, fv_raw, axis=0)
            
            balance += np.shape(fv)[0]
        
        batchData = batchData_temp[:batchSize, :]
        batchData_stats = batchData_stats_temp[:batchSize, :]
        batchLabel = batchLabel_temp[:batchSize,:]
        batchSm_pred = sm_pred_temp[:batchSize,:]
        if PARAMS['raw_feat_input']:
            batchRawFeat = raw_feat_temp[:batchSize,:]

        batchData_temp = batchData_temp[batchSize:, :]
        batchData_stats_temp = batchData_stats_temp[batchSize:, :]
        batchLabel_temp = batchLabel_temp[batchSize:, :]
        sm_pred_temp = sm_pred_temp[batchSize:,:]
        if PARAMS['raw_feat_input']:
            raw_feat_temp = raw_feat_temp[batchSize:,:]
        balance -= batchSize
                    
        batch_count += 1
        
        if add_noise:
            if PARAMS['cbow_feat_input']:
                gauss_noise = np.random.normal(loc=0.0, scale=1e-3, size=np.shape(batchData))
                batchData = np.add(batchData, gauss_noise)
            if PARAMS['sm_pred_input']:
                gauss_noise = np.random.normal(loc=0.0, scale=1e-3, size=np.shape(batchSm_pred))
                batchSm_pred = np.add(batchSm_pred, gauss_noise)
            if PARAMS['raw_feat_input']:
                gauss_noise = np.random.normal(loc=0.0, scale=1e-3, size=np.shape(batchRawFeat))
                batchRawFeat = np.add(batchRawFeat, gauss_noise)
            if PARAMS['stats_feat_input']:
                gauss_noise = np.random.normal(loc=0.0, scale=1e-3, size=np.shape(batchData_stats))
                batchData_stats = np.add(batchData_stats, gauss_noise)
                
            # batchLabel = np.append(batchLabel, batchLabel, axis=0)
        # print('Batch ', batch_count, ' shape=', np.shape(batchData), np.shape(batchLabel), np.shape(batchSm_pred))
            
        train_input = {}
        if PARAMS['raw_feat_input']:
            train_input['raw_feat_input'] = batchRawFeat
        if PARAMS['cbow_feat_input']:
            train_input['cbow_feat_input'] = batchData
        if PARAMS['sm_pred_input']:
            train_input['sm_pred_input'] = batchSm_pred
        if PARAMS['stats_feat_input']:
            train_input['stats_feat_input'] = batchData_stats

        yield train_input, batchLabel





def train_model(PARAMS, model, weightFile, logFile):
    csv_logger = CSVLogger(logFile, append=True)
    mcp = ModelCheckpoint(weightFile, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min', save_freq='epoch')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, restore_best_weights=True, min_delta=0.001, patience=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00000001, verbose=1, mode='min', min_delta=0.01)

    trainingTimeTaken = 0
    start = time.process_time()
    
    # Train the model
    print('SPE: ', PARAMS['TR_STEPS'], ' VSteps: ', PARAMS['V_STEPS'])

    model.fit(
        generator(PARAMS, PARAMS['train_files'], PARAMS['batch_size'], add_noise=PARAMS['add_noise_data_augmentation']),
        steps_per_epoch=PARAMS['TR_STEPS'],
        epochs=PARAMS['epochs'],
        verbose=1,
        validation_data=generator(PARAMS, PARAMS['val_files'], PARAMS['batch_size']),
        validation_steps=PARAMS['V_STEPS'],
        callbacks=[csv_logger, es, mcp, reduce_lr],
        )
            
    trainingTimeTaken = time.process_time() - start
    print('Time taken for model training: ',trainingTimeTaken)

    return model, trainingTimeTaken




def perform_training(PARAMS):
    modelName = '@'.join(PARAMS['modelName'].split('.')[:-1]) + '.' + PARAMS['modelName'].split('.')[-1]
    weightFile = modelName.split('.')[0] + '.h5'
    architechtureFile = modelName.split('.')[0] + '.json'
    summaryFile = modelName.split('.')[0] + '_summary.txt'
    paramFile = modelName.split('.')[0] + '_params.npz'
    logFile = modelName.split('.')[0] + '_log.csv'

    modelName = '.'.join(modelName.split('@'))
    weightFile = '.'.join(weightFile.split('@'))
    architechtureFile = '.'.join(architechtureFile.split('@'))
    summaryFile = '.'.join(summaryFile.split('@'))
    paramFile = '.'.join(paramFile.split('@'))
    logFile = '.'.join(logFile.split('@'))
    
    print('Weight file: ', weightFile)
    if not os.path.exists(paramFile):
        model, learning_rate = cnn_model(PARAMS)
        misc.print_model_summary(summaryFile, model)
        print(model.summary())
        print('Proposed 2D CNN CBoW architecture with time-feature attention')
        
        
        model, trainingTimeTaken = train_model(PARAMS, model, weightFile, logFile)
        if PARAMS['save_flag']:
            np.savez(paramFile, lr=str(learning_rate), TTT=str(trainingTimeTaken))

    model, learning_rate = cnn_model(PARAMS)
    model.load_weights(weightFile)
    print('Trained model loaded')
    trainingTimeTaken = float(np.load(paramFile)['TTT'])    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC(curve='PR')])
    
    Train_Params = {
            'model': model,
            'trainingTimeTaken': trainingTimeTaken,
            'paramFile': paramFile,
            'architechtureFile': architechtureFile,
            'weightFile': weightFile,
            }
    
    return Train_Params




def performance_metrics(PARAMS, Prediction, Groundtruth, genre_freq_test):
    # print('Predictions: ', np.shape(Prediction), ' Groundtruth:', np.shape(Groundtruth))
    P_curve = {}
    R_curve = {}
    T = {}
    AUC_values = {}
    Precision = {}
    Recall = {}
    ConfMat = {}
    F1_score = {}
    Threshold = {}
    AP = {}
    macro_avg_auc = 0
    macro_weighted_avg_auc = 0
    for genre_i in PARAMS['genre_list'].keys():
        # print(genre_i, PARAMS['genre_list'][genre_i])
        pred = Prediction[:,PARAMS['genre_list'][genre_i]]
        # print('pred: ', np.shape(pred))
        gt = Groundtruth[:,PARAMS['genre_list'][genre_i]]

        precision_curve, recall_curve, threshold = precision_recall_curve(gt, pred)
        fscore_curve = np.divide(2*np.multiply(precision_curve, recall_curve), np.add(precision_curve, recall_curve)+1e-10)
        Precision[genre_i] = np.round(np.mean(precision_curve),4)
        Recall[genre_i] = np.round(np.mean(recall_curve),4)
        F1_score[genre_i] = np.round(np.mean(fscore_curve),4)
        
        P_curve[genre_i] = precision_curve
        R_curve[genre_i] = recall_curve
        AUC_values[genre_i] = np.round(auc(recall_curve, precision_curve)*100,2)
        print('AUC: ', AUC_values[genre_i], genre_i)
        # print('AUC_values: ', AUC_values)
        macro_avg_auc += AUC_values[genre_i]
        macro_weighted_avg_auc += PARAMS['genre_freq'][genre_i]*AUC_values[genre_i]
    
    micro_avg_precision_curve, micro_avg_recall_curve, threshold = precision_recall_curve(y_true=Groundtruth.ravel(), probas_pred=Prediction.ravel())
    P_curve['micro_avg'] = micro_avg_precision_curve
    R_curve['micro_avg'] = micro_avg_recall_curve
    AUC_values['micro_avg'] = np.round(auc(micro_avg_recall_curve, micro_avg_precision_curve)*100,2)
    AUC_values['macro_avg'] = np.round(macro_avg_auc/len(PARAMS['genre_list']),2)
    AUC_values['macro_avg_weighted'] = np.round(macro_weighted_avg_auc,2)
    print('AUC (macro-avg): ', AUC_values['macro_avg'])
    print('AUC (micro-avg): ', AUC_values['micro_avg'])
    print('AUC (macro-avg weighted): ', AUC_values['macro_avg_weighted'])

    AP['macro'] = np.round(APS(y_true=Groundtruth, y_score=Prediction, average='macro')*100,2)
    AP['micro'] = np.round(APS(y_true=Groundtruth, y_score=Prediction, average='micro')*100,2)
    AP['samples'] = np.round(APS(y_true=Groundtruth, y_score=Prediction, average='samples')*100,2)
    AP['weighted'] = np.round(APS(y_true=Groundtruth, y_score=Prediction, average='weighted')*100,2)

    print('AP (macro): ', AP['macro'])
    print('AP (micro): ', AP['micro'])
    print('AP (samples): ', AP['samples'])
    print('AP (weighted): ', AP['weighted'])

    Metrics = {
        'P_curve': P_curve,
        'R_curve': R_curve,
        'threshold': T,
        'AUC': AUC_values,
        'Prediction': Prediction,
        'Groundtruth': Groundtruth,
        'Precision':Precision,
        'Recall':Recall,
        'ConfMat':ConfMat,
        'F1_score':F1_score,
        'Threshold': Threshold,
        'average_precision': AP,
        }    
    
    return Metrics





def test_model(PARAMS, Train_Params):
    testingTimeTaken = 0    
    start = time.process_time()
    
    test_files = PARAMS['test_files']
    
    Trailer_pred = np.empty([])
    Trailer_groundtruth = np.empty([])
    Segment_Pred = np.empty([])
    Segment_groundtruth = np.empty([])
    genre_freq_test = {genre_i:0 for genre_i in PARAMS['genre_list'].keys()}
    files = test_files[PARAMS['classes'][0]]
    fl_count = 0
    for fl in files:
        fl_count += 1
        if not fl.split('.')[0] in PARAMS['annotations'].keys():
            continue
        feat_fName = PARAMS['feature_path'] + '/' + fl.split('.')[0] + '.npy'
        if not os.path.exists(feat_fName):
            continue

        fv = np.load(feat_fName)
        # fv = StandardScaler().fit_transform(fv)

        sm_pred_fName = PARAMS['sp_mu_pred_path'] + '/' + fl.split('.')[0] + '.npy'
        pred_sm = np.load(sm_pred_fName)

        ''' Smoothing SM Predictions '''
        # pred_sm_smooth = np.zeros(np.shape(pred_sm))
        # pred_sm_smooth[:,1] = medfilt(pred_sm[:,1], kernel_size=PARAMS['smoothing_win_size'])
        # pred_sm_smooth[:,0] = 1 - pred_sm_smooth[:,1]
        # pred_sm = pred_sm_smooth.copy()

        stats_feat_fName = PARAMS['feature_path_Stats'] + '/' + fl.split('.')[0] + '.npy'
        fv_stats = np.load(stats_feat_fName)
        fv_stats = StandardScaler().fit_transform(fv_stats)

        lab = np.zeros(len(PARAMS['genre_list']))
        genre_strings = []
        for genre_i in PARAMS['genre_list'].keys():
            if genre_i in PARAMS['annotations'][fl.split('.')[0]]['genre']:
                lab[PARAMS['genre_list'][genre_i]] = 1
                genre_strings.append(genre_i)
                genre_freq_test[genre_i] += 1
        lab = np.repeat(np.array(lab, ndmin=2), np.shape(fv)[0], axis=0)

        # To account for trailers with no labels
        if np.sum(lab)==0:
            continue
        
        fv, lab, pred_sm, fv_stats, patch_labels = reshape_CBoW_features(PARAMS, fv, lab, pred_sm, fv_stats)

        if PARAMS['raw_feat_input']:
            fv_raw = get_raw_features(PARAMS, fl, np.shape(fv)[0])
                        
        predict_input = {}
        if PARAMS['raw_feat_input']:
            predict_input['raw_feat_input'] = fv_raw.astype(np.float32)
        if PARAMS['cbow_feat_input']:
            predict_input['cbow_feat_input'] = fv.astype(np.float32)
        if PARAMS['sm_pred_input']:
            predict_input['sm_pred_input'] = pred_sm.astype(np.float32)
        if PARAMS['stats_feat_input']:
            predict_input['stats_feat_input'] = fv_stats.astype(np.float32)
        
        Predictions = Train_Params['model'].predict(predict_input)

        # print('Predictions: ', np.shape(Predictions))

        mean_pred = np.mean(Predictions, axis=0)
        
        if np.size(Trailer_pred)<=1:
            Trailer_pred = np.array(mean_pred, ndmin=2)
            Trailer_groundtruth = np.array(lab[0,:], ndmin=2)
            Segment_Pred = Predictions
            Segment_groundtruth = np.array(lab, ndmin=2)
        else:
            Trailer_pred = np.append(Trailer_pred, np.array(mean_pred, ndmin=2), axis=0)
            Trailer_groundtruth = np.append(Trailer_groundtruth, np.array(lab[0,:], ndmin=2), axis=0)
            Segment_Pred = np.append(Segment_Pred, Predictions, axis=0)
            Segment_groundtruth = np.append(Segment_groundtruth, np.array(lab, ndmin=2), axis=0)
        
        p_curve, r_curve, threshold = precision_recall_curve(lab.ravel(), Predictions.ravel())
        ap_samples = np.round(APS(y_true=Trailer_groundtruth, y_score=Trailer_pred, average='samples')*100,2)
        print(f'({fl_count}/{len(files)}) {fl} {np.shape(fv)} Predictions: {np.shape(Predictions)} {np.shape(Trailer_pred)}, {np.shape(Trailer_groundtruth)} auc={np.round(auc(r_curve, p_curve),4)} :: AP samples={ap_samples}')

    total_genre_sum = np.sum([genre_freq_test[key] for key in genre_freq_test.keys()])
    for key in genre_freq_test.keys():
        genre_freq_test[key] /= total_genre_sum

    Segment_Metrics = performance_metrics(PARAMS, Segment_Pred, Segment_groundtruth, genre_freq_test)
    Trailer_Metrics = performance_metrics(PARAMS, Trailer_pred, Trailer_groundtruth, genre_freq_test)
    testingTimeTaken = time.process_time() - start
    Segment_Metrics['testingTimeTaken'] = testingTimeTaken
    Trailer_Metrics['testingTimeTaken'] = testingTimeTaken
    
    return Segment_Metrics, Trailer_Metrics





def get_data_mean(PARAMS):
    mean_fv = np.zeros(PARAMS['feat_dim']) #200, 300
    num_fv = 0
    num_data = 0
    genre_freq = {genre_i:0 for genre_i in PARAMS['genre_list'].keys()}

    for fl in PARAMS['train_files'][PARAMS['classes'][0]]:
        # startTime = time.process_time()
        feat_fName = PARAMS['feature_path'] + '/' + fl.split('.')[0] + '.npy'
        if not fl.split('.')[0] in PARAMS['annotations'].keys():
            print('Mean ', feat_fName, ' does not exist in Annotations')
            continue
        if not os.path.exists(PARAMS['audio_path']+'/'+fl.split('/')[-1]):
            print('Mean ', feat_fName, ' wav file does not exist')
            continue

        fv = np.load(feat_fName)

        lab = np.zeros(len(PARAMS['genre_list']))
        for genre_i in PARAMS['genre_list'].keys():
            if genre_i in PARAMS['annotations'][fl.split('.')[0]]['genre']:
                lab[PARAMS['genre_list'][genre_i]] = 1
                genre_freq[genre_i] += 1
        
        # To account for trailers with no labels
        # if np.sum(lab)==0:
        #     continue

        mean_fv = np.add(mean_fv, np.sum(fv, axis=0))        
        num_fv += np.shape(fv)[0]
        num_data += 1
        # print('mean computation: ', time.process_time()-startTime)
        
    mean_fv = np.divide(mean_fv, num_fv)
    print('mean_fv: ', np.shape(mean_fv))
    
    total_genre_sum = np.sum([genre_freq[key] for key in genre_freq.keys()])
    for key in genre_freq.keys():
        # genre_freq[key] /= total_genre_sum
        genre_freq[key] /= num_data
    print('total_genre_sum: ', total_genre_sum, num_fv, num_data)

    std_fv = np.zeros(PARAMS['feat_dim'])
    num_fv = 0
    for fl in PARAMS['train_files'][PARAMS['classes'][0]]:
        # startTime = time.process_time()
        feat_fName = PARAMS['feature_path'] + '/' + fl.split('.')[0] + '.npy'
        if not fl.split('.')[0] in PARAMS['annotations'].keys():
            print('Stdev ', feat_fName, ' does not exist in Annotations')
            continue
        if not os.path.exists(PARAMS['audio_path']+'/'+fl.split('/')[-1]):
            print('Stdev ', feat_fName, ' wav file does not exist')
            continue

        fv = np.load(feat_fName)
        
        lab = np.zeros(len(PARAMS['genre_list']))
        for genre_i in PARAMS['genre_list'].keys():
            if genre_i in PARAMS['annotations'][fl.split('.')[0]]['genre']:
                lab[PARAMS['genre_list'][genre_i]] = 1
        
        # To account for trailers with no labels
        # if np.sum(lab)==0:
        #     continue
        
        mn = np.repeat(np.array(mean_fv, ndmin=2), np.shape(fv)[0], axis=0)
        std_fv = np.add(std_fv, np.sum(np.power(np.subtract(fv,mn),2), axis=0))
        num_fv += np.shape(fv)[0]

        # print('stdev computation: ', time.process_time()-startTime)

    std_fv = np.divide(std_fv, num_fv-1)
    std_fv = np.sqrt(std_fv)
    print('std_fv: ', np.shape(std_fv))
    
    return mean_fv, std_fv, genre_freq






def print_results(PARAMS, fold, suffix, **kwargs):
    if not suffix=='':
        opFile = PARAMS['opDir'] + '/Performance_' + suffix + '.csv'
    else:
        opFile = PARAMS['opDir'] + '/Performance.csv'

    linecount = 0
    if os.path.exists(opFile):
        with open(opFile, 'r', encoding='utf8') as fid:
            for line in fid:
                linecount += 1    
    
    fid = open(opFile, 'a+', encoding = 'utf-8')
    heading = 'fold'
    values = str(fold)
    for i in range(len(kwargs)):
        heading = heading + '\t' + np.squeeze(kwargs[str(i)]).tolist().split(':')[0]
        values = values + '\t' + np.squeeze(kwargs[str(i)]).tolist().split(':')[1]

    if linecount==0:
        fid.write(heading + '\n' + values + '\n')
    else:
        fid.write(values + '\n')
        
    fid.close()




def __init__():
    PARAMS = {
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),

              # EEE-GPU
            'dataset_name': 'Moviescope',
            'feature_path':'/home1/PhD/mrinmoy.bhattacharjee/MTGC_SMO/Moviescope/1s_1s_10ms_5ms_10PT_2022-01-21/CBoW-ASPT-LSPT/',
            # # 'feature_path':'/home1/PhD/mrinmoy.bhattacharjee/MTGC_SMO/Moviescope/Native_CBoW_Features_10PT_5mix_2022-01-12/CBoW-ASPT-LSPT/',
            # # 'feature_path': '/home1/PhD/mrinmoy.bhattacharjee/MTGC_SMO/Moviescope/Native_CBoW_Features_3classes_2022-01-18/CBoW-ASPT-LSPT/',
            # 'feature_path': '/home1/PhD/mrinmoy.bhattacharjee/MTGC_SMO/Moviescope/Native_CBoW_Features_2classes_25PT_2mix_2022-02-01/CBoW-ASPT-LSPT/',
            'raw_feature_opDir':'/home1/PhD/mrinmoy.bhattacharjee/MTGC_SMO/Moviescope/Cascante_et_al_arXiv_2019/',

            'sp_mu_pred_path':'/home1/PhD/mrinmoy.bhattacharjee/MTGC_SMO/Moviescope/1s_1s_10ms_5ms_10PT_2022-01-21/SpMu_Predictions_CBoW-ASPT-LSPT/',
            # 'sp_mu_pred_path': '/home1/PhD/mrinmoy.bhattacharjee/MTGC_SMO/Moviescope/X-Vectors/SpMu_Predictions_X-Vectors/',
            
            'feature_path_Stats': '/home1/PhD/mrinmoy.bhattacharjee/MTGC_SMO/Moviescope/1s_1s_10ms_5ms_10PT_2022-01-21/Stats-ASPT-LSPT/',
            # 'feature_path_Stats':'/home1/PhD/mrinmoy.bhattacharjee/MTGC_SMO/Moviescope/Native_CBoW_Features_2classes_25PT_2mix_2022-02-01/Stats-ASPT-LSPT/',
            'audio_path':'/home1/PhD/mrinmoy.bhattacharjee/data/Moviescope/wav/',
            'annot_path':'/home1/PhD/mrinmoy.bhattacharjee/data/Moviescope/Original_Metadata.csv',
            # 'opt_hyperparam':{'num_output': 13, 'conv_lyr': 30, 'activation': 'relu', 'dropout': 0.1, 'dense_layers': 2, 'dense_nodes': 50, 'lr': 0.0001},
            # 'opt_hyperparam_MTL':{'num_output': 13, 'conv_lyr': 40, 'activation': 'relu', 'dropout': 0.1, 'dense_layers': 1, 'dense_nodes': 50, 'lr': 0.0001, 'batch_norm': False},


            # DGX-Server
            # 'dataset_name': 'Moviescope',
            # 'feature_path':'/workspace/pguha_pg/Mrinmoy/features/MTGC_SMO/Moviescope/1s_1s_10ms_5ms_10PT_2022-01-21/CBoW-ASPT-LSPT/',
            # # # 'feature_path':'/workspace/pguha_pg/Mrinmoy/features/MTGC_SMO/Moviescope/Native_CBoW_Features_2classes_2022-01-12/CBoW-ASPT-LSPT/',
            # # # 'feature_path': '/workspace/pguha_pg/Mrinmoy/features/MTGC_SMO/Moviescope/Native_CBoW_Features_3classes_2022-01-18/CBoW-ASPT-LSPT/',
            # 'raw_feature_opDir':'/workspace/pguha_pg/Mrinmoy/features/MTGC_SMO/Moviescope/Cascante_et_al_arXiv_2019/',
            # 'sp_mu_pred_path':'/workspace/pguha_pg/Mrinmoy/features/MTGC_SMO/Moviescope/1s_1s_10ms_5ms_10PT_2022-01-21/SpMu_Predictions_CBoW-ASPT-LSPT/',
            # 'feature_path_Stats': '/workspace/pguha_pg/Mrinmoy/features/MTGC_SMO/Moviescope/1s_1s_10ms_5ms_10PT_2022-01-21/Stats-ASPT-LSPT/',
            # 'audio_path':'/workspace/pguha_pg/Mrinmoy/data/Moviescope_dummy/wav/',
            # 'annot_path':'/workspace/pguha_pg/Mrinmoy/data/Moviescope_dummy/Original_Metadata.csv',
            # # 'opt_hyperparam':{'num_output': 13, 'conv_lyr': 30, 'activation': 'relu', 'dropout': 0.1, 'dense_layers': 2, 'dense_nodes': 50, 'lr': 0.0001},
            # # 'opt_hyperparam_MTL':{'num_output': 13, 'conv_lyr': 40, 'activation': 'relu', 'dropout': 0.1, 'dense_layers': 1, 'dense_nodes': 50, 'lr': 0.0001, 'batch_norm': False},
            
            
            # PARAM-ISHAN
            # 'dataset_name': 'Moviescope',
            # 'feature_path':'/scratch/mbhattacharjee/features/MTGC_SMO/Moviescope/1s_1s_10ms_5ms_10PT_2022-01-21/CBoW-ASPT-LSPT/',
            # 'raw_feature_opDir':'/scratch/mbhattacharjee/features/MTGC_SMO/Moviescope/Cascante_et_al_arXiv_2019/',
            # 'sp_mu_pred_path':'/scratch/mbhattacharjee/features/MTGC_SMO/Moviescope/1s_1s_10ms_5ms_10PT_2022-01-21/SpMu_Predictions_CBoW-ASPT-LSPT/',
            # 'feature_path_MSD':'/scratch/mbhattacharjee/features/MTGC_SMO/Moviescope/1s_1s_10ms_5ms_10PT_2022-01-21/MSD-ASPT-LSPT/',
            # 'feature_path_Stats': '/scratch/mbhattacharjee/features/MTGC_SMO/Moviescope/1s_1s_10ms_5ms_10PT_2022-01-21/Stats-ASPT-LSPT/',
            # 'audio_path':'/scratch/mbhattacharjee/data/Moviescope_dummy/wav/',
            # 'annot_path':'/scratch/mbhattacharjee/data/Moviescope_dummy/Original_Metadata.csv',
            # # 'opt_hyperparam':{'num_output': 13, 'conv_lyr': 30, 'activation': 'relu', 'dropout': 0.1, 'dense_layers': 2, 'dense_nodes': 50, 'lr': 0.0001},
            # # 'opt_hyperparam_MTL':{'num_output': 13, 'conv_lyr': 40, 'activation': 'relu', 'dropout': 0.1, 'dense_layers': 1, 'dense_nodes': 50, 'lr': 0.0001, 'batch_norm': False},



             # Lab-PC
            # 'dataset_name': 'Moviescope',
            # 'feature_path':'/home/mrinmoy/Documents/PhD_Work/Features/MTGC_SMO/Moviescope/1s_1s_10ms_5ms_10PT_2022-01-25/CBoW-ASPT-LSPT/',
            # 'raw_feature_opDir':'/home/mrinmoy/Documents/PhD_Work/Features/MTGC_SMO/Moviescope/Cascante_et_al_arXiv_2019/',
            # 'sp_mu_pred_path':'/home/mrinmoy/Documents/PhD_Work/Features/MTGC_SMO/Moviescope/1s_1s_10ms_5ms_10PT_2022-01-25/SpMu_Predictions_CBoW-ASPT-LSPT/',
            # # 'sp_mu_pred_path': '/home/mrinmoy/Documents/PhD_Work/Features/MTGC_SMO/Moviescope/X-Vectors/SpMu_Predictions_X-Vectors/',
            # 'feature_path_Stats': '/home/mrinmoy/Documents/PhD_Work/Features/MTGC_SMO/Moviescope/1s_1s_10ms_5ms_10PT_2022-01-25/Stats-ASPT-LSPT/',
            # 'audio_path':'/media/mrinmoy/NTFS_Volume/Phd_Work/Data/Moviescope/wav/',
            # 'annot_path':'/media/mrinmoy/NTFS_Volume/Phd_Work/Data/Moviescope/Original_Metadata.csv',


            
            'CV_folds': 3,
            'fold': 0,
            'save_flag': True,
            'use_GPU': True,
            'GPU_session':None,
            'classes':{0:'wav'},
            'epochs': 100,
            'batch_size': 32,
            'W': 30000, # interval size in ms
            'W_shift': 20000, # interval shift in ms
            'add_noise_data_augmentation': True,
            'feat_dim': 200,
            'raw_feat_input': False,
            'cbow_feat_input': True,
            'sm_pred_input': True,
            'stats_feat_input': True,
            'smoothing_win_size': 5,
            'possible_genres': ['Action', 'Animation', 'Biography', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller'], # Moviescope 13
            'minority_genres': ['Crime', 'Sci-Fi', 'Fantasy', 'Horror', 'Family', 'Mystery', 'Biography', 'Animation'],
            }

    PARAMS['CV_path_train'] = './cross_validation_info/' + PARAMS['dataset_name'] + '/'    
    if not os.path.exists(PARAMS['CV_path_train']+'/cv_file_list_original.pkl'):
        print('Fold division of files for cross-validation not done!')
        sys.exit(0)
    PARAMS['cv_file_list'] = misc.load_obj(PARAMS['CV_path_train'], 'cv_file_list_original')
    
    DT_SZ = 0
    for clNum in PARAMS['classes'].keys():
        classname = PARAMS['classes'][clNum]
        DT_SZ += PARAMS['cv_file_list']['total_duration'][classname] # in Hours
    DT_SZ *= 3600*1000 # in msec
    
    nTrain = len(PARAMS['cv_file_list']['original_splits']['train'])
    nVal = len(PARAMS['cv_file_list']['original_splits']['val'])
    nTest = len(PARAMS['cv_file_list']['original_splits']['test'])
    tr_frac = nTrain/(nTrain+nVal+nTest)
    vl_frac = nVal/(nTrain+nVal+nTest)
    ts_frac = nTest/(nTrain+nVal+nTest)
    PARAMS['TR_STEPS'] = int(np.floor(DT_SZ/PARAMS['W_shift'])*tr_frac/(PARAMS['batch_size']))
    PARAMS['V_STEPS'] = int(np.floor(DT_SZ/PARAMS['W_shift'])*vl_frac/(PARAMS['batch_size']))
    PARAMS['TS_STEPS'] = int(np.floor(DT_SZ/PARAMS['W_shift'])*ts_frac/(PARAMS['batch_size']))
    print('TR_STEPS: %d, \tV_STEPS: %d, \tTS_STEPS: %d\n'%(PARAMS['TR_STEPS'], PARAMS['V_STEPS'], PARAMS['TS_STEPS']))

    return PARAMS



if __name__ == '__main__':
    PARAMS = __init__()
    PARAMS['annotations'], PARAMS['genre_list'] = misc.get_annotations(PARAMS['annot_path'], PARAMS['possible_genres'])
    print(PARAMS['genre_list'])
    # sys.exit(0)
    
    suffix = 'Attention'
    if PARAMS['cbow_feat_input']:
        suffix += '_CBoW'
    if PARAMS['sm_pred_input']:
        # suffix += '_XVector-SMPred'
        suffix += '_SMPred'
    if PARAMS['stats_feat_input']:
        suffix += '_Stats'
    if PARAMS['raw_feat_input']:
        suffix += '_LMS'
    suffix += '_STL_' + str(int(PARAMS['W']/1000)) + 's_NoAttn'
        
    PARAMS['opDir'] = './results/' + PARAMS['dataset_name'] + '/' + PARAMS['today'] + '/' + suffix + '/'
    if not os.path.exists(PARAMS['opDir']):
        os.makedirs(PARAMS['opDir'])

    PARAMS['train_files'] = {'wav':PARAMS['cv_file_list']['original_splits']['train']}
    PARAMS['val_files'] = {'wav':PARAMS['cv_file_list']['original_splits']['val']}
    PARAMS['test_files'] = {'wav':PARAMS['cv_file_list']['original_splits']['test']}
    print('train_files: ', PARAMS['train_files'])
    print('test_files: ', PARAMS['test_files'])

    misc.print_configuration(PARAMS)

    if PARAMS['use_GPU']:
        PARAMS['GPU_session'] = start_GPU_session()

    if not os.path.exists(PARAMS['opDir']+'/data_stats_fold' + str(PARAMS['fold']) + '.pkl'):
        PARAMS['mean_fv'], PARAMS['std_fv'], PARAMS['genre_freq'] = get_data_mean(PARAMS)
        misc.save_obj({
            'mean_fv':PARAMS['mean_fv'],
            'std_fv':PARAMS['std_fv'], 
            'genre_freq':PARAMS['genre_freq']
            }, PARAMS['opDir'], 'data_stats_fold' + str(PARAMS['fold']))
    else:
        PARAMS['mean_fv'] = misc.load_obj(PARAMS['opDir'], 'data_stats_fold' + str(PARAMS['fold']))['mean_fv']
        PARAMS['std_fv'] = misc.load_obj(PARAMS['opDir'], 'data_stats_fold' + str(PARAMS['fold']))['std_fv']
        PARAMS['genre_freq'] = misc.load_obj(PARAMS['opDir'], 'data_stats_fold' + str(PARAMS['fold']))['genre_freq']
        
    print('mean_fv loaded: ', np.shape(PARAMS['mean_fv']))
    print('std_fv loaded: ', np.shape(PARAMS['std_fv']))
    print('genre_freq: ', PARAMS['genre_freq'])

    PARAMS['genre_weights'] = {}
    max_freq = np.max([PARAMS['genre_freq'][genre_i] for genre_i in PARAMS['genre_freq'].keys()])
    for genre_i in PARAMS['genre_freq'].keys():
        lab = PARAMS['genre_list'][genre_i]
        PARAMS['genre_weights'][lab] = max_freq/PARAMS['genre_freq'][genre_i]
    print('genre_weights: ', PARAMS['genre_weights'].keys())
    print(PARAMS['genre_weights'])
    
    
    PARAMS['modelName'] = PARAMS['opDir'] + '/fold' + str(PARAMS['fold']) + '_model.xyz'
    Train_Params = perform_training(PARAMS)

    Segment_Metrics = {}
    Trailer_Metrics = {}
    if not os.path.exists(PARAMS['opDir']+'/Test_Params_fold'+str(PARAMS['fold'])+'.pkl'):
        Segment_Metrics, Trailer_Metrics = test_model(PARAMS, Train_Params)
        misc.save_obj({'Segment_Metrics':Segment_Metrics, 'Trailer_Metrics':Trailer_Metrics}, PARAMS['opDir'], 'Test_Params_fold'+str(PARAMS['fold']))
    else:
        Segment_Metrics = misc.load_obj(PARAMS['opDir'], 'Test_Params_fold'+str(PARAMS['fold']))['Segment_Metrics']
        Trailer_Metrics = misc.load_obj(PARAMS['opDir'], 'Test_Params_fold'+str(PARAMS['fold']))['Trailer_Metrics']

    for genre_i in PARAMS['genre_list'].keys():
        # print(genre_i, 'AUC=', Segment_Metrics['AUC'][genre_i])
        result = {
            '0':'genre:'+genre_i,
            '1':'AUC (macro):'+str(Segment_Metrics['AUC']['macro_avg']),
            '2':'AUC (micro):'+str(Segment_Metrics['AUC']['micro_avg']),
            '3':'AUC (weighted):'+str(Segment_Metrics['AUC']['macro_avg_weighted']),
            '4':'AUC:'+str(Segment_Metrics['AUC'][genre_i]),
            '5':'Precision:'+str(Segment_Metrics['Precision'][genre_i]),
            '6':'Recall:'+str(Segment_Metrics['Recall'][genre_i]),
            '7':'F1_score:'+str(Segment_Metrics['F1_score'][genre_i]),
            '8':'AP (macro):'+str(Segment_Metrics['average_precision']['macro']),
            '9':'AP (micro):'+str(Segment_Metrics['average_precision']['micro']),
            '10':'AP (samples):'+str(Segment_Metrics['average_precision']['samples']),
            '11':'AP (weighted):'+str(Segment_Metrics['average_precision']['weighted']),
            }
        print_results(PARAMS, PARAMS['fold'], 'segment_level', **result)

    for genre_i in PARAMS['genre_list'].keys():
        # print(genre_i, 'AUC=', Trailer_Metrics['AUC'][genre_i])
        result = {
            '0':'genre:'+genre_i,
            '1':'AUC (macro):'+str(Trailer_Metrics['AUC']['macro_avg']),
            '2':'AUC (micro):'+str(Trailer_Metrics['AUC']['micro_avg']),
            '3':'AUC (weighted):'+str(Trailer_Metrics['AUC']['macro_avg_weighted']),
            '4':'AUC:'+str(Trailer_Metrics['AUC'][genre_i]),
            '5':'Precision:'+str(Trailer_Metrics['Precision'][genre_i]),
            '6':'Recall:'+str(Trailer_Metrics['Recall'][genre_i]),
            '7':'F1_score:'+str(Trailer_Metrics['F1_score'][genre_i]),
            '8':'AP (macro):'+str(Trailer_Metrics['average_precision']['macro']),
            '9':'AP (micro):'+str(Trailer_Metrics['average_precision']['micro']),
            '10':'AP (samples):'+str(Trailer_Metrics['average_precision']['samples']),
            '11':'AP (weighted):'+str(Trailer_Metrics['average_precision']['weighted']),
            }
        print_results(PARAMS, PARAMS['fold'], 'trailer_level', **result)

    if PARAMS['use_GPU']:
        reset_TF_session()
        