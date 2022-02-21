#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 12:28:56 2022

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 02:33:12 2022

@author: mrinmoy
"""

import numpy as np
np.random.seed(1989)
import lib.misc as misc
import os
import datetime
import librosa
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, BatchNormalization, Input, Dropout, Activation, Conv2D, MaxPooling2D, Concatenate, Cropping2D, LSTM, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.metrics import AUC
import time
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import average_precision_score as APS
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





def CRNN_model(PARAMS, weightFile):
    # tf.keras.backend.clear_session()
    
    # create model
    
    if PARAMS['num_model_parameters']=='0.1M':
        # 0.1M parameters: 30-60-60-60-30-30
        layer_width = [30, 60, 60, 60, 30, 30]
    if PARAMS['num_model_parameters']=='0.25M':
        layer_width = [48, 96, 96, 96, 48, 48]
    if PARAMS['num_model_parameters']=='0.5M':
        layer_width = [68, 137, 137, 137, 68, 68]
    if PARAMS['num_model_parameters']=='1M':
        layer_width = [96, 195, 195, 195, 96, 96]
    elif PARAMS['num_model_parameters']=='3M':
        # 3M parameters: 169-339-339-339-169-169
        layer_width = [169, 339, 339, 339, 169, 169]
    
    input_layer = Input(shape=(128,1407,1)) # (None, 128, 1407, 1)
    
    x = Conv2D(layer_width[0], kernel_size=(3,3), strides=(1,1), padding='same')(input_layer) # (None, 128, 1407, 16) , kernel_regularizer=l2()
    x = BatchNormalization(axis=-1)(x)
    # x = Activation('elu')(x)
    # x = Dropout(0.5)(x)
    
    x = MaxPooling2D(pool_size=(2,2), padding='same')(x) # (None, 64, 704, N_channels)
    
    x = Conv2D(layer_width[1], kernel_size=(3,3), strides=(1,1), padding='same')(x) # (None, 64, 704, N_channels)
    x = BatchNormalization(axis=-1)(x)
    # x = Activation('elu')(x)
    # x = Dropout(0.5)(x)
    
    x = MaxPooling2D(pool_size=(3,3), padding='same')(x) # (None, 22, 235, N_channels)
    
    x = Conv2D(layer_width[2], kernel_size=(3,3), strides=(1,1), padding='same')(x) # (None, 22, 235, N_channels)
    x = BatchNormalization(axis=-1)(x)
    # x = Activation('elu')(x)
    # x = Dropout(0.5)(x)
    
    x = MaxPooling2D(pool_size=(4,4), padding='same')(x) # (None, 6, 59, N_channels)
    
    x = Conv2D(layer_width[3], kernel_size=(3,3), strides=(1,1), padding='valid')(x) # (None, 4, 57, N_channels)
    x = BatchNormalization(axis=-1)(x)
    # x = Activation('elu')(x)
    # x = Dropout(0.5)(x)
    
    x = MaxPooling2D(pool_size=(4,4), padding='same')(x) # (None, 1, 15, N_channels)

    x = tf.squeeze(x, axis=1) # (None, 15, N_channels) time x channels
    
    x = LSTM(layer_width[4], return_sequences=True)(x) # (None, 15, N_recurrent)
    x = BatchNormalization(axis=-1)(x)
    
    x = LSTM(layer_width[5], return_sequences=True)(x) # (None, 15, N_recurrent)
    x = BatchNormalization(axis=-1)(x)
    
    x = Flatten()(x)
    
    output_lyr = Dense(13, activation='sigmoid')(x)

    model = Model(input_layer, output_lyr)
    model.load_weights(weightFile)
    opt = optimizers.Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[AUC(curve='PR')])

    return model




def compute_features(PARAMS, fName):
    startTime = time.process_time()
    Xin, fs = librosa.load(PARAMS['audio_path']+'/'+fName.split('/')[-1], mono=True, sr=12000)
    duration = np.round(len(Xin) / float(fs),2)
    print(fName, ' Xin: ', np.shape(Xin), ' fs=', fs, f'duration = {duration} seconds')
    print('wav reading: ', time.process_time()-startTime)
    
    startTime = time.process_time()
    Xin -= np.mean(Xin)
    Xin /= (np.max(Xin)-np.min(Xin))
    print('standardizing: ', time.process_time()-startTime)
    
    startTime = time.process_time()
    MelSpec = librosa.feature.melspectrogram(y=Xin, sr=fs, n_fft=2048, hop_length=PARAMS['hop_length'], center=False, power=2.0, n_mels=PARAMS['n_mels'])
    LogMelSpec = librosa.power_to_db(MelSpec)
    print('log mel spec computing: ', time.process_time()-startTime)
    
    startTime = time.process_time()
    nFrames = np.shape(LogMelSpec)[1]
    seg_size = int(PARAMS['W']/(PARAMS['hop_length']*1000/fs))+1
    print('LogMelSpec: ', np.shape(LogMelSpec), nFrames, seg_size)
    if nFrames<seg_size:
        LogMelSpec_temp = LogMelSpec.copy()
        while np.shape(LogMelSpec_temp)[1]<seg_size:
            LogMelSpec_temp = np.append(LogMelSpec_temp, LogMelSpec, axis=1)
        LogMelSpec = LogMelSpec_temp.copy()
        nFrames = np.shape(LogMelSpec)[1]
        print('LogMelSpec (appended): ', np.shape(LogMelSpec), nFrames, seg_size)
    print('size increaing: ', time.process_time()-startTime)
    
    startTime = time.process_time()
    FV = np.empty([])
    frmStart = 0
    frmEnd = 0
    for seg_i in range(PARAMS['nSeg']):
        frmEnd = np.min([frmStart+seg_size, nFrames])
        if (frmEnd-frmStart)<seg_size:
            frmStart = np.random.randint(nFrames-seg_size)
            frmEnd = np.min([frmStart+seg_size, nFrames])
        if np.size(FV)<=1:
            FV = np.expand_dims(LogMelSpec[:,frmStart:frmEnd], axis=0)
        else:
            FV = np.append(FV, np.expand_dims(LogMelSpec[:,frmStart:frmEnd], axis=0), axis=0)
        frmStart = frmEnd
    print('segment extraction: ', time.process_time()-startTime)

    print('FV: ', np.shape(FV))
    
    return FV




def generator_filewise_test(PARAMS, fName):        
    feat_fName = PARAMS['raw_feature_opDir'] + '/' + fName.split('.')[0] + '.pkl'
    if not os.path.exists(feat_fName):
        fv = compute_features(PARAMS, fName)
        misc.save_obj(fv, PARAMS['raw_feature_opDir'], fName.split('.')[0])
    else:
        fv = misc.load_obj(PARAMS['raw_feature_opDir'], fName.split('.')[0])

    fv_reshaped = np.empty([])
    for seg_i in range(np.shape(fv)[0]):
        if np.size(fv_reshaped)<=1:
            fv_reshaped = fv[seg_i, :]
        else:
            fv_reshaped = np.append(fv_reshaped, fv[seg_i,:], axis=1)
    fv_reshaped = fv_reshaped.T
    fv_reshaped = StandardScaler().fit_transform(fv_reshaped)
    fv_scaled = np.empty([])
    frmStart = 0
    for seg_i in range(np.shape(fv)[0]):
        if np.size(fv_scaled)<=1:
            fv_scaled = np.expand_dims(fv_reshaped[frmStart:frmStart+1407, :].T, axis=0)
        else:
            fv_scaled = np.append(fv_scaled, np.expand_dims(fv_reshaped[frmStart:frmStart+1407,:].T, axis=0), axis=0)
        frmStart += 1407

    fv_scaled = np.expand_dims(fv_scaled, axis=3)
    # print('test: ', fName, np.shape(fv_scaled))
    
    test_label = np.zeros(len(PARAMS['Moviescope_genres']))
    for genre_i in PARAMS['genre_list'].keys():
        if genre_i in PARAMS['annotations'][fName.split('.')[0]]['genre']:
            test_label[PARAMS['EmoGDB_genres'][genre_i]] = 1
    test_label = np.repeat(np.array(test_label, ndmin=2), np.shape(fv_scaled)[0], axis=0)

    return fv_scaled, test_label




def test_model_filewise(PARAMS, trained_model):
    start = time.process_time()
    # loss, performance 

    test_files = PARAMS['test_files']
    
    Predictions = np.empty([])
    test_label = np.empty([])
    genre_freq_test = {genre_i:0 for genre_i in PARAMS['genre_list'].keys()}
    fl_count = 0
    files = test_files['wav']
    for fl in files:
        fl_count += 1
        if not fl.split('.')[0] in PARAMS['annotations'].keys():
            continue
        if not os.path.exists(PARAMS['audio_path']+'/'+fl.split('/')[-1]):
            continue

        test_data = None
        labels = None
        test_data, labels = generator_filewise_test(PARAMS, fl)
        if np.sum(labels)==0:
            continue
        for genre_i in PARAMS['genre_list'].keys():
            if genre_i in PARAMS['annotations'][fl.split('.')[0]]['genre']:
                genre_freq_test[genre_i] += 1
        
        pred = None
        pred = trained_model.predict(test_data)

        pred = np.mean(pred, axis=0)
        labels = labels[0,:] # np.mean(labels, axis=0)
        if np.size(Predictions)<=1:
            Predictions = np.array(pred, ndmin=2)
            test_label = np.array(labels, ndmin=2)
        else:
            Predictions = np.append(Predictions, np.array(pred, ndmin=2), axis=0)
            test_label = np.append(test_label, np.array(labels, ndmin=2), axis=0)

        p_curve, r_curve, threshold = precision_recall_curve(labels.ravel(), pred.ravel())
        ap_samples = np.round(APS(y_true=test_label, y_score=Predictions, average='samples')*100,2)
        print(f'({fl_count}/{len(files)}) {fl} {np.shape(test_data)} Predictions: {np.shape(pred)} {np.shape(Predictions)}, {np.shape(test_label)} auc={np.round(auc(r_curve, p_curve),4)} :: AP samples={ap_samples}')

    total_genre_sum = np.sum([genre_freq_test[key] for key in genre_freq_test.keys()])
    for key in genre_freq_test.keys():
        genre_freq_test[key] /= total_genre_sum
            
    print('Trailer_pred: ', np.shape(Predictions), np.shape(test_label))
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
    for genre_i in PARAMS['EmoGDB_genres'].keys():
        # print(genre_i, PARAMS['genre_list'][genre_i])
        pred = Predictions[:,PARAMS['EmoGDB_genres'][genre_i]]
        # print('pred: ', np.shape(pred))
        gt = test_label[:,PARAMS['EmoGDB_genres'][genre_i]]

        precision_curve, recall_curve, threshold = precision_recall_curve(gt, pred)
        fscore_curve = np.divide(2*np.multiply(precision_curve, recall_curve), np.add(precision_curve, recall_curve)+1e-10)
        Precision[genre_i] = np.round(np.mean(precision_curve),4)
        Recall[genre_i] = np.round(np.mean(recall_curve),4)
        F1_score[genre_i] = np.round(np.mean(fscore_curve),4)

        P_curve[genre_i] = precision_curve
        R_curve[genre_i] = recall_curve
        AUC_values[genre_i] = np.round(auc(recall_curve, precision_curve)*100,2)
        macro_avg_auc += AUC_values[genre_i]
        macro_weighted_avg_auc += genre_freq_test[genre_i]*AUC_values[genre_i]

    micro_avg_precision_curve, micro_avg_recall_curve, threshold = precision_recall_curve(test_label.ravel(), Predictions.ravel())
    P_curve['micro_avg'] = micro_avg_precision_curve
    R_curve['micro_avg'] = micro_avg_recall_curve
    AUC_values['micro_avg'] = np.round(auc(micro_avg_recall_curve, micro_avg_precision_curve)*100,2)
    AUC_values['macro_avg'] = np.round(macro_avg_auc/len(PARAMS['genre_list']),2)
    AUC_values['macro_avg_weighted'] = np.round(macro_weighted_avg_auc,2)
    print('AUC (macro-avg): ', AUC_values['macro_avg'])
    print('AUC (micro-avg): ', AUC_values['micro_avg'])
    print('AUC (macro-avg weighted): ', AUC_values['macro_avg_weighted'])

    emogdb_labels = [PARAMS['EmoGDB_genres'][key] for key in PARAMS['EmoGDB_genres'].keys()]
    Predictions = Predictions[:, emogdb_labels]
    test_label = test_label[:, emogdb_labels]
    print(f'Predictions: {np.shape(Predictions)}, Groundtruth: {np.shape(test_label)}')
    AP['macro'] = np.round(APS(y_true=test_label, y_score=Predictions, average='macro')*100,2)
    AP['micro'] = np.round(APS(y_true=test_label, y_score=Predictions, average='micro')*100,2)
    AP['samples'] = np.round(APS(y_true=test_label, y_score=Predictions, average='samples')*100,2)
    AP['weighted'] = np.round(APS(y_true=test_label, y_score=Predictions, average='weighted')*100,2)

    print('AP (macro): ', AP['macro'])
    print('AP (micro): ', AP['micro'])
    print('AP (samples): ', AP['samples'])
    print('AP (weighted): ', AP['weighted'])
    
    testingTimeTaken = time.process_time() - start

    Test_Params = {
        'testingTimeTaken': testingTimeTaken, 
        'P_curve': P_curve,
        'R_curve': R_curve,
        'threshold': T,
        'AUC': AUC_values,
        'Predictions': Predictions,
        'test_label': test_label,
        'Precision':Precision,
        'Recall':Recall,
        'ConfMat':ConfMat,
        'F1_score':F1_score,
        'Threshold': Threshold,
        'average_precision': AP,
        }    
    
    return Test_Params





def cnn_model(PARAMS, weightFile):    
    nConv_cbow = 80
    nConv_stats = 80
    nConv_raw = 50
    act = 'relu'
    do = 0.1
    n_fc_nodes = 300
    n_fc_lyrs = 3
    num_output = 13
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
        
        try:
            x = Concatenate(axis=-1)([x, x_sm])
        except:
            x = x_sm


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
    model.load_weights(weightFile)
    opt = optimizers.Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[AUC(curve='PR')])
    
    return model




def load_model(PARAMS):
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
    if PARAMS['method']=='proposed':
        model = cnn_model(PARAMS, weightFile)
    elif PARAMS['method']=='baseline':
        model = CRNN_model(PARAMS, weightFile)
    
    Train_Params = {
            'model': model,
            # 'trainingTimeTaken': trainingTimeTaken,
            'paramFile': paramFile,
            'architechtureFile': architechtureFile,
            'weightFile': weightFile,
            }
    
    return Train_Params





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
    emogdb_labels = [PARAMS['EmoGDB_genres'][key] for key in PARAMS['EmoGDB_genres'].keys()]
    Prediction = Prediction[:, emogdb_labels]
    # Prediction = np.divide(Prediction, np.repeat(np.array(np.max(Prediction, axis=1), ndmin=2).T, np.shape(Prediction)[1], axis=1))
    Groundtruth = Groundtruth[:, emogdb_labels]
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
        macro_weighted_avg_auc += genre_freq_test[genre_i]*AUC_values[genre_i]
    
    micro_avg_precision_curve, micro_avg_recall_curve, threshold = precision_recall_curve(y_true=Groundtruth.ravel(), probas_pred=Prediction.ravel())
    P_curve['micro_avg'] = micro_avg_precision_curve
    R_curve['micro_avg'] = micro_avg_recall_curve
    AUC_values['micro_avg'] = np.round(auc(micro_avg_recall_curve, micro_avg_precision_curve)*100,2)
    AUC_values['macro_avg'] = np.round(macro_avg_auc/len(PARAMS['genre_list']),2)
    AUC_values['macro_avg_weighted'] = np.round(macro_weighted_avg_auc,2)
    print('AUC (macro-avg): ', AUC_values['macro_avg'])
    print('AUC (micro-avg): ', AUC_values['micro_avg'])
    print('AUC (macro-avg weighted): ', AUC_values['macro_avg_weighted'])
    
    print(f'Prediction: {np.shape(Prediction)}, Groundtruth: {np.shape(Groundtruth)}')
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

        lab = np.zeros(len(PARAMS['Moviescope_genres']))
        genre_strings = []
        for genre_i in PARAMS['genre_list'].keys():
            if genre_i in PARAMS['annotations'][fl.split('.')[0]]['genre']:
                lab[PARAMS['EmoGDB_genres'][genre_i]] = 1
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
        else:
            Trailer_pred = np.append(Trailer_pred, np.array(mean_pred, ndmin=2), axis=0)
            Trailer_groundtruth = np.append(Trailer_groundtruth, np.array(lab[0,:], ndmin=2), axis=0)
        
        p_curve, r_curve, threshold = precision_recall_curve(lab.ravel(), Predictions.ravel())
        ap_samples = np.round(APS(y_true=Trailer_groundtruth, y_score=Trailer_pred, average='samples')*100,2)
        print(f'({fl_count}/{len(files)}) {fl} {np.shape(fv)} Predictions: {np.shape(Predictions)} {np.shape(Trailer_pred)}, {np.shape(Trailer_groundtruth)} auc={np.round(auc(r_curve, p_curve),4)} :: AP samples={ap_samples}')

    total_genre_sum = np.sum([genre_freq_test[key] for key in genre_freq_test.keys()])
    for key in genre_freq_test.keys():
        genre_freq_test[key] /= total_genre_sum

    Trailer_Metrics = performance_metrics(PARAMS, Trailer_pred, Trailer_groundtruth, genre_freq_test)
    testingTimeTaken = time.process_time() - start
    Trailer_Metrics['testingTimeTaken'] = testingTimeTaken
    
    return Trailer_Metrics





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

             # Laptop
            'dataset_name': 'EmoGDB',
            'model_path_proposed': './results/Moviescope/Best/Proposed_2DCNN_Attention/Attention_CBoW_SMPred_Stats/', # CBoW+SMPred+Stats
            'model_path_baseline':'./results/Moviescope/Best/Cascante_et_al_arXiv_2019',
            'feature_path':'./features/EmoGDB/1s_1s_10ms_5ms_10PT_2022-02-03/CBoW-ASPT-LSPT/',
            'raw_feature_opDir':'./features/EmoGDB/Cascante_et_al_arXiv_2019/',
            'sp_mu_pred_path':'./features/EmoGDB/1s_1s_10ms_5ms_10PT_2022-02-03/SpMu_Predictions_CBoW-ASPT-LSPT/',
            'feature_path_Stats': './features/EmoGDB/1s_1s_10ms_5ms_10PT_2022-02-03/Stats-ASPT-LSPT/',
            'audio_path':'/media/mrinmoy/NTFS_Volume/Phd_Work/Data/EmoGDB/wav/',
            'annot_path':'/media/mrinmoy/NTFS_Volume/Phd_Work/Data/EmoGDB/Annotations.csv',
            
            'method':'proposed', # 'proposed', 'baseline'
            'CV_folds': 3,
            'fold': 0,
            'save_flag': True,
            'use_GPU': False,
            'GPU_session':None,
            'classes':{0:'wav'},
            'epochs': 100,
            'batch_size': 32,
            'W': 30000, # interval size in ms
            'W_shift': 30000, # interval shift in ms
            'feat_dim': 200,
            'raw_feat_input': False,
            'cbow_feat_input': True,
            'sm_pred_input': True,
            'stats_feat_input': True,
            'smoothing_win_size': 5,
            'hop_length': 256, # hop-length in samples
            'n_mels': 128, # Number of Mel filters
            'nSeg': 4, # Number of 30s chunks from a trailer
            'num_model_parameters': '0.1M', #0.1M, 0.25M, 0.5M, 1M, 3M
            'possible_genres': ['Action', 'Animation', 'Biography', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller'], # Moviescope 13
            'Moviescope_genres': {'Action': 0, 'Fantasy': 1, 'Sci-Fi': 2, 'Thriller': 3, 'Romance': 4, 'Animation': 5, 'Comedy': 6, 'Family': 7, 'Mystery': 8, 'Drama': 9, 'Crime': 10, 'Horror': 11, 'Biography': 12},
            'EmoGDB_genres': {'Action':0, 'Thriller':3, 'Romance':4, 'Comedy':6, 'Drama':9, 'Horror':11}
            }

    return PARAMS



if __name__ == '__main__':
    PARAMS = __init__()
    PARAMS['annotations'], PARAMS['genre_list'] = misc.get_annotations(PARAMS['annot_path'], PARAMS['possible_genres'])
    print(PARAMS['genre_list'])
    # import sys
    # sys.exit(0)
    
    suffix = 'CrossDataset'
    if PARAMS['method']=='baseline':
        suffix += '_Cascante_et_al'
    else:
        suffix += '_Attention'
        if PARAMS['cbow_feat_input']:
            suffix += '_CBoW'
        if PARAMS['sm_pred_input']:
            suffix += '_SMPred'
        if PARAMS['stats_feat_input']:
            suffix += '_Stats'
        if PARAMS['raw_feat_input']:
            suffix += '_LMS'
    suffix += '_STL_' + str(int(PARAMS['W']/1000)) + 's_' + PARAMS['method']
        
    PARAMS['opDir'] = './results/' + PARAMS['dataset_name'] + '/' + PARAMS['today'] + '/' + suffix + '/'
    if not os.path.exists(PARAMS['opDir']):
        os.makedirs(PARAMS['opDir'])

    PARAMS['test_files'] = {'wav':[fl.split('/')[-1] for fl in librosa.util.find_files(PARAMS['audio_path'], ext=['wav'])]}
    print('test_files: ', PARAMS['test_files'])

    misc.print_configuration(PARAMS)

    if PARAMS['use_GPU']:
        PARAMS['GPU_session'] = start_GPU_session()

    PARAMS['modelName'] = PARAMS['model_path_'+str(PARAMS['method'])] + '/fold' + str(PARAMS['fold']) + '_model.xyz'
    Train_Params = load_model(PARAMS)

    Trailer_Metrics = {}
    if not os.path.exists(PARAMS['opDir']+'/Test_Params_fold'+str(PARAMS['fold'])+'.pkl'):
        if PARAMS['method']=='proposed':
            Trailer_Metrics = test_model(PARAMS, Train_Params)
        elif PARAMS['method']=='baseline':
            Trailer_Metrics = test_model_filewise(PARAMS, Train_Params['model'])
        misc.save_obj({'Trailer_Metrics':Trailer_Metrics}, PARAMS['opDir'], 'Test_Params_fold'+str(PARAMS['fold']))
    else:
        Trailer_Metrics = misc.load_obj(PARAMS['opDir'], 'Test_Params_fold'+str(PARAMS['fold']))['Trailer_Metrics']

    for genre_i in PARAMS['EmoGDB_genres'].keys():
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
        
