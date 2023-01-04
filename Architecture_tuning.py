#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 00:06:29 2022

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
np.random.seed(1989)
import os
import datetime
import lib.misc as misc
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, BatchNormalization, Input, Dropout, Activation, Conv2D, MaxPooling2D, Concatenate, Cropping2D
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau
import time
import sys
import keras_tuner as kt
from keras_tuner import HyperParameters, BayesianOptimization, RandomSearch
import io
from contextlib import redirect_stdout            
from lib.attention_layer import MrinSelfAttention as attn_lyr
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import AUC
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import average_precision_score as APS
from tensorflow.keras.regularizers import l2
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
    tf.random.set_seed(1989)




def reset_TF_session():
    tf.compat.v1.keras.backend.clear_session()




def cnn_model(hp):
    nConv_cbow = hp.get('nConv_cbow')
    nConv_stats = hp.get('nConv_stats')
    act = hp.get('activation')
    do = hp.get('dropout')
    n_fc_nodes = hp.get('dense_nodes')
    n_fc_lyrs = hp.get('dense_layers')
    learning_rate = 0.001
    
    
    cbow_input = Input(shape=(200,30,1), name='cbow_feat_input') # (None, 200, 30, 1)
    
    ''' CBoW-ASPT '''
    x_cbow_aspt = Cropping2D(cropping=((0, 100), (0, 0)))(cbow_input) # (None, 100, 30, 1)

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
    x_cbow_lspt = Cropping2D(cropping=((100, 0), (0, 0)))(cbow_input) # (None, 100, 30, 1)

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
    x = Concatenate(axis=-1)([x_cbow_time, x_cbow_feat]) # (None, 4*nConv+30)


    stats_feat_input = Input(shape=(240,30,1), name='stats_feat_input')

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

    x = Concatenate(axis=-1)([x, x_stats])


    sm_pred_input = Input((2,30,), name='sm_pred_input')
    x_sm = K.permute_dimensions(sm_pred_input, [0,2,1])  # (None, 30, 2)
    
    x_sm_time = attn_lyr(attention_dim=1)(x_sm, reduce_sum=True) # (None, 30, 2)
    x_sm_time = BatchNormalization(axis=-1)(x_sm_time)
    x_sm_feat = attn_lyr(attention_dim=2)(x_sm, reduce_sum=True) # (None, 30, 2)
    x_sm_feat = BatchNormalization(axis=-1)(x_sm_feat)
    x_sm = Concatenate(axis=-1)([x_sm_time, x_sm_feat])
    
    x = Concatenate(axis=-1)([x, x_sm])


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
    inputs['cbow_feat_input'] = cbow_input
    inputs['sm_pred_input'] = sm_pred_input
    inputs['stats_feat_input'] = stats_feat_input

    output_layer = Dense(13, activation='sigmoid')(x)        
    model = Model(inputs, output_layer)
    opt = optimizers.Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[AUC(curve='PR')])
    
    return model

    
    


def get_tuner(PARAMS):
    hp = HyperParameters()

    hp.Int('nConv_cbow', min_value=70, max_value=90, step=1)
    hp.Int('nConv_stats', min_value=70, max_value=90, step=1)
    hp.Choice('activation', ['relu', 'elu'])
    hp.Float('dropout', min_value=0.0, max_value=0.4, step=0.05)
    hp.Int('dense_nodes', min_value=50, max_value=500, step=50)
    hp.Int('dense_layers', min_value=1, max_value=5, step=1)
        
    tuner = RandomSearch(
        cnn_model,
        hyperparameters = hp,
        objective = kt.Objective('val_auc', direction='max'),
        max_trials = PARAMS['max_trials'],
        executions_per_trial = 1,
        overwrite = False,
        directory = PARAMS['opDir'],
        project_name = 'CBoW_SMPred_Stats',
        tune_new_entries = True,
        allow_new_entries = True,
        )
    
    return tuner




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
                                                    
            if np.size(batchData_temp)<=1:
                batchData_temp = fv
                batchData_stats_temp = fv_stats
                batchLabel_temp = lab
                sm_pred_temp = pred_sm
            else:
                batchData_temp = np.append(batchData_temp, fv, axis=0)
                batchData_stats_temp = np.append(batchData_stats_temp, fv_stats, axis=0)
                batchLabel_temp = np.append(batchLabel_temp, lab, axis=0)
                sm_pred_temp = np.append(sm_pred_temp, pred_sm, axis=0)
            
            balance += np.shape(fv)[0]
        
        batchData = batchData_temp[:batchSize, :]
        batchData_stats = batchData_stats_temp[:batchSize, :]
        batchLabel = batchLabel_temp[:batchSize,:]
        batchSm_pred = sm_pred_temp[:batchSize,:]

        batchData_temp = batchData_temp[batchSize:, :]
        batchData_stats_temp = batchData_stats_temp[batchSize:, :]
        batchLabel_temp = batchLabel_temp[batchSize:, :]
        sm_pred_temp = sm_pred_temp[batchSize:,:]
        balance -= batchSize
                    
        batch_count += 1
        
        batchData = np.add(batchData, np.random.normal(loc=0.0, scale=1e-3, size=np.shape(batchData)))
        batchSm_pred = np.add(batchSm_pred, np.random.normal(loc=0.0, scale=1e-3, size=np.shape(batchSm_pred)))
        batchData_stats = np.add(batchData_stats, np.random.normal(loc=0.0, scale=1e-3, size=np.shape(batchData_stats)))
                
            # batchLabel = np.append(batchLabel, batchLabel, axis=0)
        # print('Batch ', batch_count, ' shape=', np.shape(batchData), np.shape(batchLabel), np.shape(batchSm_pred))
            
        train_input = {}
        train_input['cbow_feat_input'] = batchData
        train_input['sm_pred_input'] = batchSm_pred
        train_input['stats_feat_input'] = batchData_stats

        if PARAMS['model_type']=='STL':
            yield train_input, batchLabel

        elif PARAMS['model_type']=='MTL':
            targets = {}
            genre_list_labels = {PARAMS['genre_list'][genre_i]:genre_i for genre_i in PARAMS['genre_list'].keys()}
            for genre_lab in range(len(PARAMS['genre_list'])):
                targets[genre_list_labels[genre_lab]] = batchLabel[:, genre_lab]
                
            # print('Batch ', batch_count, ' shape=', np.shape(batchData), np.shape(batchLabel), np.shape(batchSm_pred), np.shape(targets))
            
            yield train_input, targets





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
    
    if PARAMS['model_type']=='STL':
        loss, auc_measure = Train_Params['model'].evaluate(
            generator(PARAMS, test_files, PARAMS['batch_size']), 
            steps=PARAMS['TS_STEPS'],
            )
    elif PARAMS['model_type']=='MTL':
        eval_res = Train_Params['model'].evaluate(
            generator(PARAMS, test_files, PARAMS['batch_size']), 
            steps=PARAMS['TS_STEPS'],
            )
        print('eval_res: ', eval_res)
        print(Train_Params['model'].metrics_names)
        loss = 0
        auc_measure = 0
        for genre_i in PARAMS['genre_list'].keys():
            loss_idx = np.squeeze(np.where([genre_i+'_loss' in metric for metric in Train_Params['model'].metrics_names]))
            auc_idx = np.squeeze(np.where([genre_i+'_auc' in metric for metric in Train_Params['model'].metrics_names]))
            loss += eval_res[loss_idx]
            auc_measure += eval_res[auc_idx]
        loss = np.round(loss, 4)
        auc_measure = np.round(auc_measure/len(PARAMS['genre_list']), 4)
    print('evaluation: ', loss, auc_measure)
    
    Trailer_pred = np.empty([])
    Trailer_groundtruth = np.empty([])
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

        predict_input = {}
        predict_input['cbow_feat_input'] = fv
        predict_input['sm_pred_input'] = pred_sm
        predict_input['stats_feat_input'] = fv_stats
        
        if PARAMS['model_type']=='STL':
            Predictions = Train_Params['model'].predict(predict_input)
        elif PARAMS['model_type']=='MTL':
            Predictions_MTL = Train_Params['model'].predict(predict_input)
            Predictions = np.zeros((np.shape(predict_input['cbow_feat_input'])[0], len(PARAMS['genre_list'])))
            for genre_lab in range(len(PARAMS['genre_list'])):
                Predictions[:,genre_lab] = np.array(Predictions_MTL[genre_lab], ndmin=2).T

        # print('Predictions: ', np.shape(Predictions))
        p_curve, r_curve, threshold = precision_recall_curve(lab.ravel(), Predictions.ravel())

        mean_pred = np.mean(Predictions, axis=0)
        mean_lab = np.mean(lab, axis=0)
        
        if np.size(Trailer_pred)<=1:
            Trailer_pred = np.array(mean_pred, ndmin=2)
            Trailer_groundtruth = np.array(mean_lab, ndmin=2)
        else:
            Trailer_pred = np.append(Trailer_pred, np.array(mean_pred, ndmin=2), axis=0)
            Trailer_groundtruth = np.append(Trailer_groundtruth, np.array(mean_lab, ndmin=2), axis=0)
            
        print(f'({fl_count}/{len(files)}) {fl} Predictions: {np.shape(Trailer_pred)}, {np.shape(Trailer_groundtruth)} auc={np.round(auc(r_curve, p_curve),4)}')

    total_genre_sum = np.sum([genre_freq_test[key] for key in genre_freq_test.keys()])
    for key in genre_freq_test.keys():
        genre_freq_test[key] /= total_genre_sum

    Trailer_Metrics = performance_metrics(PARAMS, Trailer_pred, Trailer_groundtruth, genre_freq_test)
    testingTimeTaken = time.process_time() - start
    Trailer_Metrics['testingTimeTaken'] = testingTimeTaken
    
    return Trailer_Metrics





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
            # 'feature_path':'/home1/PhD/mrinmoy.bhattacharjee/MTGC_SMO/Moviescope/Native_CBoW_Features_10PT_5mix_2022-01-12/CBoW-ASPT-LSPT/',
            # 'feature_path': '/home1/PhD/mrinmoy.bhattacharjee/MTGC_SMO/Moviescope/Native_CBoW_Features_3classes_2022-01-18/CBoW-ASPT-LSPT/',
            'raw_feature_opDir':'/home1/PhD/mrinmoy.bhattacharjee/MTGC_SMO/Moviescope/Cascante_et_al_arXiv_2019/',
            'sp_mu_pred_path':'/home1/PhD/mrinmoy.bhattacharjee/MTGC_SMO/Moviescope/1s_1s_10ms_5ms_10PT_2022-01-21/SpMu_Predictions_CBoW-ASPT-LSPT/',
            'feature_path_Stats': '/home1/PhD/mrinmoy.bhattacharjee/MTGC_SMO/Moviescope/1s_1s_10ms_5ms_10PT_2022-01-21/Stats-ASPT-LSPT/',
            'audio_path':'/home1/PhD/mrinmoy.bhattacharjee/data/Moviescope/wav/',
            'annot_path':'/home1/PhD/mrinmoy.bhattacharjee/data/Moviescope/Original_Metadata.csv',
            
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
            'model_type': 'STL', # 'STL', 'MTL'
            'smoothing_win_size': 5,
            'max_trials': 50,
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

    PARAMS['opDir'] = './results/' + PARAMS['dataset_name'] + '/Architecture_tuning/Proposed_MTGC_2DCNN_Attention_Model_' + PARAMS['today'] + '/'
    if not os.path.exists(PARAMS['opDir']):
        os.makedirs(PARAMS['opDir'])

    PARAMS['annotations'], PARAMS['genre_list'] = misc.get_annotations(PARAMS['annot_path'], PARAMS['possible_genres'])

    PARAMS['fold'] = 0
    # Original split
    PARAMS['train_files'] = {'wav':PARAMS['cv_file_list']['original_splits']['train']}
    PARAMS['val_files'] = {'wav':PARAMS['cv_file_list']['original_splits']['val']}
    PARAMS['test_files'] = {'wav':PARAMS['cv_file_list']['original_splits']['test']}
    print('train_files: ', PARAMS['train_files'])
    print('test_files: ', PARAMS['test_files'])

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

    if PARAMS['use_GPU']:
        PARAMS['GPU_session'] = start_GPU_session()
            
    tuner_obj = get_tuner(PARAMS)
    start_time = time.process_time()
    print('SPE: ', PARAMS['TR_STEPS'], ' ValSteps: ', PARAMS['V_STEPS'])

    csv_logger = CSVLogger(PARAMS['opDir']+'/tuner_search_log.csv', append=True)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, restore_best_weights=True, min_delta=0.001, patience=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00000001, verbose=1, mode='min', min_delta=0.01)
    
    tuner_obj.search(
        generator(PARAMS, PARAMS['train_files'], PARAMS['batch_size']),
        steps_per_epoch=PARAMS['TR_STEPS'],
        epochs=PARAMS['epochs'],
        verbose=1,
        validation_data=generator(PARAMS, PARAMS['val_files'], PARAMS['batch_size']),
        validation_steps=PARAMS['V_STEPS'],
        callbacks=[es, csv_logger, reduce_lr],
        # max_queue_size=2,
        workers=1
        )
    
    weightFile = PARAMS['opDir'] + '/best_model.h5'
    architechtureFile = PARAMS['opDir'] + '/best_model.json'
    paramFile = PARAMS['opDir'] + '/best_model.npz'

    best_model = tuner_obj.get_best_models(1)[0]
    trainingTimeTaken = time.process_time() - start_time

    best_model.save_weights(weightFile) # Save the weights
    with open(architechtureFile, 'w') as f: # Save the model architecture
        f.write(best_model.to_json())
    np.savez(paramFile, epochs=PARAMS['epochs'], batch_size=PARAMS['batch_size'], trainingTimeTaken=trainingTimeTaken)

    Trailer_Metrics = test_model(PARAMS, {'model':best_model})
    print('Trailer_Metrics: ', Trailer_Metrics)

    for genre_i in PARAMS['genre_list'].keys():
        print(genre_i, 'AUC=', Trailer_Metrics['AUC'][genre_i])
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
        print_results(PARAMS, PARAMS['fold'], '', **result)

    f = io.StringIO()
    with redirect_stdout(f):
        tuner_obj.results_summary()
    summary = f.getvalue()
    with open(PARAMS['opDir']+'/tuner_results.txt', 'w+') as f:
        f.write(summary)

    best_hparams = tuner_obj.get_best_hyperparameters(1)[0].values
    print(best_hparams)
    with open(PARAMS['opDir']+'/best_hyperparameters.csv', 'w+') as f:
        for key in best_hparams.keys():
            f.write(key + ',' + str(best_hparams[key]) + '\n')

    if PARAMS['use_GPU']:
        reset_TF_session()

