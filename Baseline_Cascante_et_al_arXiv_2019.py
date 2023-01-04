#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 23:03:06 2021

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati

@reference: 
    P. Cascante-Bonilla, K. Sitaraman, M. Luo, and V. Ordonez, 
    “Moviescope: Large-scale analysis of movies using multiple modalities,” 
    arXiv preprint arXiv:1908.03180, 2019.
    
Implementation details:
    1. Genres: 
        action, animation, biography, comedy, crime, drama, family, fantasy, 
        horror, mystery, romance, sci-fi, and thriller
    2. 4 audio clips of 30s from each trailer. For trailers less than 2min, 
    remaining clips are extracted randomly
    3. Audio is downsampled to 12kHz. Log-Mel power spectrogram is computed for 
    each clip with 128 Mel filters, hop-size of 256 samples (21ms). 
    Matrix size=128 x 1407. Output of convolution subnetwork is N x 1 x 15 
    (number of feature maps x frequency x time)
    4. Frame-size is not mentioned. It appears the authors used default value in 
    the Librosa library of 2048 samples.
    5. For classification, following CRNN model was used:
        K. Choi, G. Fazekas, M. Sandler and K. Cho, "Convolutional recurrent 
        neural networks for music classification," 2017 IEEE International 
        Conference on Acoustics, Speech and Signal Processing (ICASSP), 2017, 
        pp. 2392-2396, doi: 10.1109/ICASSP.2017.7952585.
    6. 100 epochs, 32 batch size
    7. Results reported as area under the precision recall curve

"""

import numpy as np
import os
import datetime
import lib.misc as misc
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, BatchNormalization, Input, Dropout, Conv2D, MaxPooling2D, Activation,  LSTM, Flatten
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, Callback, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
import time
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
import sys
import librosa
import gc
from sklearn.preprocessing import StandardScaler



class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        # K.clear_session()



def start_GPU_session():
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.6)
    config = tf.compat.v1.ConfigProto(
        device_count={'GPU': 1 , 'CPU': 1}, 
        gpu_options=gpu_options,
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        )
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    tf.compat.v1.disable_eager_execution()  





def reset_TF_session():
    tf.compat.v1.keras.backend.clear_session()




def CRNN_model(PARAMS):
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
    
    output_lyr = Dense(len(PARAMS['genre_list']), activation='sigmoid')(x)

    model = Model(input_layer, output_lyr)
    
    opt = optimizers.Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[AUC(curve='PR')])

    return model





def generator(PARAMS, file_list, batchSize):
    batch_count = 0

    file_list_temp = file_list[PARAMS['classes'][0]].copy()
    np.random.shuffle(file_list_temp)

    batchData_temp = np.empty([], dtype=float)
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
            feat_fName = PARAMS['feature_opDir'] + '/' + fName.split('.')[0] + '.pkl'

            # startTime = time.process_time()
            if not os.path.exists(feat_fName):
                fv = compute_features(PARAMS, fName)
                misc.save_obj(fv, PARAMS['feature_opDir'], fName.split('.')[0])
                # print(fName, ' Feat computing time: ', np.round(time.process_time()-startTime,2))
            else:
                fv = misc.load_obj(PARAMS['feature_opDir'], fName.split('.')[0])
                # print(fName, ' Feat loading time: ', np.round(time.process_time()-startTime,2))
                            
            fv_reshaped = np.empty([])
            for seg_i in range(np.shape(fv)[0]):
                if np.size(fv_reshaped)<=1:
                    fv_reshaped = fv[seg_i, :]
                else:
                    fv_reshaped = np.append(fv_reshaped, fv[seg_i,:], axis=1)
            # print('fv_reshaped: ', np.shape(fv_reshaped), np.shape(fv))
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
            
            
            lab = np.zeros(len(PARAMS['genre_list']))
            for genre_i in PARAMS['genre_list'].keys():
                if genre_i in PARAMS['annotations'][fName.split('.')[0]]['genre']:
                    lab[PARAMS['genre_list'][genre_i]] = 1
                
            # To account for trailers with no genre labels
            if np.sum(lab)==0:
                # Augmenting minority genres
                for genre_i in PARAMS['genre_list'].keys():
                    if genre_i in PARAMS['minority_genres']:
                        lab[PARAMS['genre_list'][genre_i]] = 1
                # continue
            lab = np.repeat(np.array(lab, ndmin=2), np.shape(fv_scaled)[0], axis=0)
            
            if np.size(batchData_temp)<=1:
                batchData_temp = fv_scaled
                batchLabel_temp = lab
            else:
                batchData_temp = np.append(batchData_temp, fv_scaled, axis=0)
                batchLabel_temp = np.append(batchLabel_temp, lab, axis=0)
            
            balance += np.shape(fv_scaled)[0]
        
        batchData = batchData_temp[:batchSize, :]
        batchLabel = batchLabel_temp[:batchSize,:]

        batchData = np.expand_dims(batchData, axis=3)

        batchData_temp = batchData_temp[batchSize:, :]
        batchLabel_temp = batchLabel_temp[batchSize:, :]
        balance -= batchSize
                    
        batch_count += 1
        # print('Batch ', batch_count, ' shape=', np.shape(batchData), np.shape(batchLabel))

        add_noise = True
        if add_noise:
            # print('Adding noise data augmentation')
            gauss_noise = np.random.normal(loc=0.0, scale=1e-3, size=np.shape(batchData))
            batchData_noisy = batchData.copy()
            batchData_noisy = np.add(batchData_noisy, gauss_noise)
            batchData = np.append(batchData, batchData_noisy, axis=0)
            batchLabel = np.append(batchLabel, batchLabel, axis=0)
        
        yield batchData, batchLabel




def train_model(PARAMS, model, weightFile, logFile):
    csv_logger = CSVLogger(logFile, append=True)
    mcp = ModelCheckpoint(weightFile, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min', save_freq='epoch')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, restore_best_weights=True, min_delta=0.01, patience=20)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.000001, verbose=1, mode='min', min_delta=0.01)    
    trainingTimeTaken = 0
    start = time.process_time()
    
    # Train the model
    SPE = np.max([int(len(PARAMS['train_files'][PARAMS['classes'][0]]*4)/PARAMS['batch_size']), 1])
    VSteps = np.max([int(len(PARAMS['val_files'][PARAMS['classes'][0]])*4/PARAMS['batch_size']), 1])
    print('SPE: ', SPE, ' VSteps: ', VSteps)

    model.fit(
        generator(PARAMS, PARAMS['train_files'], PARAMS['batch_size']),
        steps_per_epoch=SPE,
        epochs=PARAMS['epochs'],
        verbose=1,
        validation_data=generator(PARAMS, PARAMS['val_files'], PARAMS['batch_size']),
        validation_steps=VSteps,
        callbacks=[csv_logger, es, mcp, reduce_lr],
        class_weight=PARAMS['genre_weights'],
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
        model = CRNN_model(PARAMS)
        misc.print_model_summary(summaryFile, model)
        print(model.summary())
        print('Architecture of Cascante et al., arXiv, 2019')
        
        model, trainingTimeTaken = train_model(PARAMS, model, weightFile, logFile)
        if PARAMS['save_flag']:
            np.savez(paramFile, TTT=str(trainingTimeTaken), learning_rate_decay=0.001)

    trainingTimeTaken = float(np.load(paramFile)['TTT'])
    model = CRNN_model(PARAMS)
    model.load_weights(weightFile)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC(curve='PR')])

    print('Cascante et al. model exists! Loaded. Training time required=',trainingTimeTaken)
    # print(model.summary())
    
    Train_Params = {
            'model': model,
            'trainingTimeTaken': trainingTimeTaken,
            # 'learning_rate_decay': learning_rate_decay,
            'paramFile': paramFile,
            'architechtureFile': architechtureFile,
            'weightFile': weightFile,
            }
    
    return Train_Params




def generator_filewise_test(PARAMS, fName):        
    feat_fName = PARAMS['feature_opDir'] + '/' + fName.split('.')[0] + '.pkl'
    if not os.path.exists(feat_fName):
        fv = compute_features(PARAMS, fName)
        misc.save_obj(fv, PARAMS['feature_opDir'], fName.split('.')[0])
    else:
        fv = misc.load_obj(PARAMS['feature_opDir'], fName.split('.')[0])

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
    
    test_label = np.zeros(len(PARAMS['genre_list']))
    for genre_i in PARAMS['genre_list'].keys():
        if genre_i in PARAMS['annotations'][fName.split('.')[0]]['genre']:
            test_label[PARAMS['genre_list'][genre_i]] = 1
    test_label = np.repeat(np.array(test_label, ndmin=2), np.shape(fv_scaled)[0], axis=0)

    return fv_scaled, test_label




def test_model_filewise(PARAMS, trained_model):
    start = time.process_time()
    # loss, performance 

    test_files = PARAMS['test_files']
    
    eval_result_fName = PARAMS['opDir'] + '/eval_result_fold' + str(PARAMS['fold']) + '.pkl'
    if not os.path.exists(eval_result_fName):

        eval_steps = int(len(test_files['wav'])*4/PARAMS['batch_size'])
        eval_loss, eval_auc = trained_model.evaluate(
            generator(PARAMS, test_files, PARAMS['batch_size']), 
            steps=eval_steps,
            )
        print('evaluation: ', eval_loss, eval_auc)
        
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
            p_curve, r_curve, threshold = precision_recall_curve(labels.ravel(), pred.ravel())
    
            pred = np.mean(pred, axis=0)
            labels = labels[0,:] # np.mean(labels, axis=0)
            if np.size(Predictions)<=1:
                Predictions = np.array(pred, ndmin=2)
                test_label = np.array(labels, ndmin=2)
            else:
                Predictions = np.append(Predictions, np.array(pred, ndmin=2), axis=0)
                test_label = np.append(test_label, np.array(labels, ndmin=2), axis=0)
    
            print(f'({fl_count}/{len(files)}) {fl} Predictions: {np.shape(Predictions)}, {np.shape(test_label)} auc={np.round(auc(r_curve, p_curve),4)}')
    
        total_genre_sum = np.sum([genre_freq_test[key] for key in genre_freq_test.keys()])
        for key in genre_freq_test.keys():
            genre_freq_test[key] /= total_genre_sum
        
        misc.save_obj({'Predictions': Predictions, 'test_label':test_label, 'eval_loss':eval_loss, 'eval_auc':eval_auc, 'genre_freq_test': genre_freq_test}, PARAMS['opDir'], 'eval_result_fold' + str(PARAMS['fold']))
    else:
        Predictions = misc.load_obj(PARAMS['opDir'], 'eval_result_fold' + str(PARAMS['fold']))['Predictions']
        test_label = misc.load_obj(PARAMS['opDir'], 'eval_result_fold' + str(PARAMS['fold']))['test_label']
        eval_loss = misc.load_obj(PARAMS['opDir'], 'eval_result_fold' + str(PARAMS['fold']))['eval_loss']
        eval_auc = misc.load_obj(PARAMS['opDir'], 'eval_result_fold' + str(PARAMS['fold']))['eval_auc']
        genre_freq_test = misc.load_obj(PARAMS['opDir'], 'eval_result_fold' + str(PARAMS['fold']))['genre_freq_test']
        
    
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
    macro_avg_auc = 0
    macro_weighted_avg_auc = 0
    for genre_i in PARAMS['genre_list'].keys():
        # print(genre_i, PARAMS['genre_list'][genre_i])
        pred = Predictions[:,PARAMS['genre_list'][genre_i]]
        # print('pred: ', np.shape(pred))
        gt = test_label[:,PARAMS['genre_list'][genre_i]]

        precision_curve, recall_curve, threshold = precision_recall_curve(gt, pred)
        fscore_curve = np.divide(2*np.multiply(precision_curve, recall_curve), np.add(precision_curve, recall_curve)+1e-10)
        Precision[genre_i] = np.round(np.mean(precision_curve),4)
        Recall[genre_i] = np.round(np.mean(recall_curve),4)
        F1_score[genre_i] = np.round(np.mean(fscore_curve),4)

        P_curve[genre_i] = precision_curve
        R_curve[genre_i] = recall_curve
        AUC_values[genre_i] = np.round(auc(recall_curve, precision_curve)*100,2)
        macro_avg_auc += AUC_values[genre_i]
        macro_weighted_avg_auc += PARAMS['genre_freq'][genre_i]*AUC_values[genre_i]

    micro_avg_precision_curve, micro_avg_recall_curve, threshold = precision_recall_curve(test_label.ravel(), Predictions.ravel())
    P_curve['micro_avg'] = micro_avg_precision_curve
    R_curve['micro_avg'] = micro_avg_recall_curve
    AUC_values['micro_avg'] = np.round(auc(micro_avg_recall_curve, micro_avg_precision_curve)*100,2)
    AUC_values['macro_avg'] = np.round(macro_avg_auc/len(PARAMS['genre_list']),2)
    AUC_values['macro_avg_weighted'] = np.round(macro_weighted_avg_auc,2)
    print('AUC (macro-avg): ', AUC_values['macro_avg'])
    print('AUC (micro-avg): ', AUC_values['micro_avg'])
    print('AUC (macro-avg weighted): ', AUC_values['macro_avg_weighted'])

    AP = {}
    AP['macro'] = np.round(average_precision_score(y_true=test_label, y_score=Predictions, average='macro')*100,2)
    AP['micro'] = np.round(average_precision_score(y_true=test_label, y_score=Predictions, average='micro')*100,2)
    AP['samples'] = np.round(average_precision_score(y_true=test_label, y_score=Predictions, average='samples')*100,2)
    AP['weighted'] = np.round(average_precision_score(y_true=test_label, y_score=Predictions, average='weighted')*100,2)

    print('AP (macro): ', AP['macro'])
    print('AP (micro): ', AP['micro'])
    print('AP (weighted): ', AP['weighted'])
    print('AP (samples): ', AP['samples'])
    
    testingTimeTaken = time.process_time() - start

    Test_Params = {
        'testingTimeTaken': testingTimeTaken, 
        'eval_results': [np.round(eval_loss,4), np.round(eval_auc,4)],
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



def get_train_test_files(PARAMS):
    train_files = {}
    test_files = {}
    for clNum in PARAMS['classes'].keys():
        class_name = PARAMS['classes'][clNum]
        train_files[class_name] = []
        test_files[class_name] = []
        for i in range(PARAMS['CV_folds']):
            files = PARAMS['cv_file_list'][class_name]['fold'+str(i)]
            if PARAMS['fold']==i:
                test_files[class_name].extend(files)
            else:
                train_files[class_name].extend(files)
    
    return train_files, test_files




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




def get_data_mean(PARAMS):
    frame_sum = np.zeros(128)
    mean_frame = np.zeros(128)
    nFrames = 0
    num_data = 0
    genre_freq = {genre_i:0 for genre_i in PARAMS['genre_list'].keys()}

    for fl in PARAMS['train_files'][PARAMS['classes'][0]]:
        # startTime = time.process_time()
        feat_fName = PARAMS['feature_opDir'] + '/' + fl.split('.')[0] + '.pkl'
        if not fl.split('.')[0] in PARAMS['annotations'].keys():
            print('Mean ', feat_fName, ' does not exist in Annotations')
            continue
        if not os.path.exists(PARAMS['audio_path']+'/'+fl.split('/')[-1]):
            print('Mean ', feat_fName, ' wav file does not exist')
            continue

        if not os.path.exists(feat_fName):
            fv = compute_features(PARAMS, fl)
            misc.save_obj(fv, PARAMS['feature_opDir'], fl.split('.')[0])
            # print('Mean Saved: ', feat_fName)
        else:
            fv = misc.load_obj(PARAMS['feature_opDir'], fl.split('.')[0])
            # print('Mean fv Loaded: ', feat_fName)
        
        lab = np.zeros(len(PARAMS['genre_list']))
        for genre_i in PARAMS['genre_list'].keys():
            if genre_i in PARAMS['annotations'][fl.split('.')[0]]['genre']:
                lab[PARAMS['genre_list'][genre_i]] = 1
                genre_freq[genre_i] += 1
        
        # To account for trailers with no labels
        # if np.sum(lab)==0:
        #     continue
        
        for seg_i in range(np.shape(fv)[0]):
            frame_sum = np.add(frame_sum, np.sum(fv[seg_i,:], axis=1))
        nFrames += 4*1407
        num_data += 1
        # print('mean computation: ', time.process_time()-startTime)

    mean_frame = np.divide(frame_sum, nFrames)
    for key in genre_freq.keys():
        genre_freq[key] /= num_data

    frame_diff = np.zeros(128)
    std_frame = np.zeros(128)
    for fl in PARAMS['train_files'][PARAMS['classes'][0]]:
        # startTime = time.process_time()
        feat_fName = PARAMS['feature_opDir'] + '/' + fl.split('.')[0] + '.pkl'
        if not fl.split('.')[0] in PARAMS['annotations'].keys():
            print('Stdev ', feat_fName, ' does not exist in Annotations')
            continue
        if not os.path.exists(PARAMS['audio_path']+'/'+fl.split('/')[-1]):
            print('Stdev ', feat_fName, ' wav file does not exist')
            continue
        
        if not os.path.exists(feat_fName):
            fv = compute_features(PARAMS, fl)
            misc.save_obj(fv, PARAMS['feature_opDir'], fl.split('.')[0])
            # print('Std Saved: ', feat_fName)
        else:
            fv = misc.load_obj(PARAMS['feature_opDir'], fl.split('.')[0])
            # print('Std fv Loaded: ', feat_fName)

        lab = np.zeros(len(PARAMS['genre_list']))
        for genre_i in PARAMS['genre_list'].keys():
            if genre_i in PARAMS['annotations'][fl.split('.')[0]]['genre']:
                lab[PARAMS['genre_list'][genre_i]] = 1

        # To account for trailers with no labels
        # if np.sum(lab)==0:
        #     continue

        for seg_i in range(np.shape(fv)[0]):
            sqrd_diff = np.sum(np.power(np.subtract(fv[seg_i,:], np.repeat(np.array(mean_frame, ndmin=2).T, 1407, axis=1)),2), axis=1)
            frame_diff = np.add(frame_diff, sqrd_diff)
        # print('stdev computation: ', time.process_time()-startTime)

    std_frame = np.divide(frame_diff, nFrames-1)
    std_frame = np.sqrt(std_frame)
    
    return mean_frame, std_frame, genre_freq




def print_results(PARAMS, fold, **kwargs):
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
            # 'dataset_name': 'EmoGDB',
            # 'feature_path': './features/',
            # 'audio_path': '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/EmoGDB/wav/',
            # 'annot_path': '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/EmoGDB/Annotations_6genre.csv',

            # EEE-GPU
            # 'dataset_name': 'EmoGDB',
            # 'feature_path':'/home1/PhD/mrinmoy.bhattacharjee/MTGC_SMO/',
            # 'audio_path':'/home/phd/mrinmoy.bhattacharjee/data/EmoGDB/wav/',
            # 'annot_path':'/home/phd/mrinmoy.bhattacharjee/data/EmoGDB/Annotations_6genre.csv',

            # EEE-GPU
            # 'dataset_name': 'movie-trailers-dataset-master',
            # 'feature_path':'/home1/PhD/mrinmoy.bhattacharjee/MTGC_SMO/',
            # 'audio_path':'/home/phd/mrinmoy.bhattacharjee/data/movie-trailers-dataset-master/wav/',
            # 'annot_path':'/home/phd/mrinmoy.bhattacharjee/data/movie-trailers-dataset-master/Annotations.csv',

             # EEE-GPU
            'dataset_name': 'Moviescope',
            'feature_path':'/home1/PhD/mrinmoy.bhattacharjee/MTGC_SMO/',
            'audio_path':'/home1/PhD/mrinmoy.bhattacharjee/data/Moviescope/wav/',
            'annot_path':'/home1/PhD/mrinmoy.bhattacharjee/data/Moviescope/Original_Metadata.csv',
            
            # LabPC
            # 'dataset_name': 'Moviescope',
            # 'feature_path': '/home/mrinmoy/Documents/PhD_Work/Features/MTGC_SMO/',
            # 'audio_path': '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/Moviescope/wav/',
            # 'annot_path': '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/Moviescope/Annotations_13genres.csv',
            
            # LabPC
            # 'dataset_name': 'movie-trailers-dataset-master',
            # 'feature_path': '/home/mrinmoy/Documents/PhD_Work/Features/MTGC_SMO/',
            # 'audio_path': '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/movie-trailers-dataset-master/wav/',
            # 'annot_path': '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/movie-trailers-dataset-master/Annotations_6genre.csv',
            
             # DGX-CSE
            # 'dataset_name': 'Moviescope',
            # 'feature_path':'/workspace/pguha_pg/Mrinmoy/MTGC_SMO/',
            # 'audio_path':'/workspace/pguha_pg/Mrinmoy/data/Moviescope_dummy/wav/',
            # 'annot_path':'/workspace/pguha_pg/Mrinmoy/data/Moviescope_dummy/Annotations_13genres.csv',

            'CV_folds': 3,
            'fold': 0,
            'save_flag': True,
            'use_GPU': True,
            'GPU_session':None,
            'classes':{0:'wav'},
            'epochs': 100,
            'batch_size': 32,
            'W': 30000, # interval size in ms
            'W_shift': 30000, # interval shift in ms
            'hop_length': 256, # hop-length in samples
            'n_mels': 128, # Number of Mel filters
            'nSeg': 4, # Number of 30s chunks from a trailer
            'num_model_parameters': '0.1M', #0.1M, 0.25M, 0.5M, 1M, 3M
            # 'possible_genres': ['Action', 'Fantasy', 'Sci-Fi', 'Thriller', 'Romance', 'Family', 'Mystery', 'Comedy', 'Drama', 'Animation', 'Crime', 'Horror', 'Biography', 'Adventure', 'Music', 'War', 'History', 'Sport', 'Musical', 'Documentary', 'Western', 'Film-Noir', 'Short', 'News', 'Reality-TV', 'Game-Show'], # Moviescope all
            'possible_genres': ['Action', 'Animation', 'Biography', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller'], # Moviescope 13
            'minority_genres': ['Crime', 'Sci-Fi', 'Fantasy', 'Horror', 'Family', 'Mystery', 'Biography', 'Animation'],
            }

    PARAMS['CV_path_train'] = './cross_validation_info/' + PARAMS['dataset_name'] + '/'    
    if not os.path.exists(PARAMS['CV_path_train']+'/cv_file_list_original.pkl'):
        print('Fold division of files for cross-validation not done!')
        sys.exit(0)
    PARAMS['cv_file_list'] = misc.load_obj(PARAMS['CV_path_train'], 'cv_file_list_original')
    
    return PARAMS



if __name__ == '__main__':
    PARAMS = __init__()
    
    PARAMS['feature_opDir'] = PARAMS['feature_path'] + '/' + PARAMS['dataset_name'] + '/Cascante_et_al_arXiv_2019/'
    if not os.path.exists(PARAMS['feature_opDir']):
        os.makedirs(PARAMS['feature_opDir'])

    PARAMS['opDir'] = './results/' + PARAMS['dataset_name'] + '/' + PARAMS['today'] + '/Cascante_et_al_arXiv_2019/'
    if not os.path.exists(PARAMS['opDir']):
        os.makedirs(PARAMS['opDir'])

    PARAMS['annotations'], PARAMS['genre_list'] = misc.get_annotations(PARAMS['annot_path'], PARAMS['possible_genres'])
    
    all_fold_auc = {genre_i:[] for genre_i in PARAMS['genre_list'].keys()}

    all_fold_auc_micro_avg = []
    all_fold_auc_macro_avg = []
    all_fold_auc_macro_avg_weighted = []

    all_fold_AP_macro = []
    all_fold_AP_micro = []
    all_fold_AP_samples = []
    all_fold_AP_weighted = []

    all_fold_prec = {genre_i:[] for genre_i in PARAMS['genre_list'].keys()}
    all_fold_rec = {genre_i:[] for genre_i in PARAMS['genre_list'].keys()}
    all_fold_fscore = {genre_i:[] for genre_i in PARAMS['genre_list'].keys()}

    for PARAMS['fold'] in range(1): # Original split
        PARAMS['train_files'] = {'wav':PARAMS['cv_file_list']['original_splits']['train']}
        PARAMS['val_files'] = {'wav':PARAMS['cv_file_list']['original_splits']['val']}
        PARAMS['test_files'] = {'wav':PARAMS['cv_file_list']['original_splits']['test']}
        print('train_files: ', PARAMS['train_files'])
        print('test_files: ', PARAMS['test_files'])

        if PARAMS['use_GPU']:
            PARAMS['GPU_session'] = start_GPU_session()
        
        if not os.path.exists(PARAMS['opDir']+'/data_stats_fold' + str(PARAMS['fold']) + '.pkl'):
            PARAMS['mean_frame'], PARAMS['std_frame'], PARAMS['genre_freq'] = get_data_mean(PARAMS)
            misc.save_obj({
                'mean_frame':PARAMS['mean_frame'],
                'std_frame':PARAMS['std_frame'], 
                'genre_freq':PARAMS['genre_freq']
                }, PARAMS['opDir'], 'data_stats_fold' + str(PARAMS['fold']))
        else:
            PARAMS['mean_frame'] = misc.load_obj(PARAMS['opDir'], 'data_stats_fold' + str(PARAMS['fold']))['mean_frame']
            PARAMS['std_frame'] = misc.load_obj(PARAMS['opDir'], 'data_stats_fold' + str(PARAMS['fold']))['std_frame']
            PARAMS['genre_freq'] = misc.load_obj(PARAMS['opDir'], 'data_stats_fold' + str(PARAMS['fold']))['genre_freq']
            
        print('mean_frame loaded: ', np.shape(PARAMS['mean_frame']))
        print('std_frame loaded: ', np.shape(PARAMS['std_frame']))
        print('genre_freq: ', PARAMS['genre_freq'])
        
        mean_nan = np.isnan(PARAMS['mean_frame'])
        stdev_nan = np.isnan(PARAMS['std_frame'])
        print(f'mean NaN: {np.sum(mean_nan)}, stdev NaN: {np.sum(stdev_nan)}')
        
        PARAMS['genre_weights'] = {}
        max_freq = np.max([PARAMS['genre_freq'][genre_i] for genre_i in PARAMS['genre_freq'].keys()])
        for genre_i in PARAMS['genre_freq'].keys():
            lab = PARAMS['genre_list'][genre_i]
            PARAMS['genre_weights'][lab] = max_freq/PARAMS['genre_freq'][genre_i]
        print('genre_weights: ', PARAMS['genre_weights'].keys())
        print(PARAMS['genre_weights'])

        PARAMS['modelName'] = PARAMS['opDir'] + '/fold' + str(PARAMS['fold']) + '_model.xyz'
        Train_Params = perform_training(PARAMS)

        if not os.path.exists(PARAMS['opDir']+'/Test_Params_fold'+str(PARAMS['fold'])+'.pkl'):
            Test_Params = test_model_filewise(PARAMS, Train_Params['model'])
            misc.save_obj(Test_Params, PARAMS['opDir'], 'Test_Params_fold'+str(PARAMS['fold']))
        else:
            Test_Params = misc.load_obj(PARAMS['opDir'], 'Test_Params_fold'+str(PARAMS['fold']))
        
        print('Test keys: ', Test_Params['AUC'].keys())
        for genre_i in PARAMS['genre_list'].keys():
            print(genre_i, 'AUC=', Test_Params['AUC'][genre_i])
            kwargs = {
                '0':'genre:'+genre_i,
                '1':'eval_loss:'+str(Test_Params['eval_results'][0]),
                '2':'eval_auc:'+str(Test_Params['eval_results'][1]),
                '3':'AUC (macro):'+str(Test_Params['AUC']['macro_avg']),
                '4':'AUC (micro):'+str(Test_Params['AUC']['micro_avg']),
                '5':'AUC (weighted):'+str(Test_Params['AUC']['macro_avg_weighted']),
                '6':'AUC:'+str(Test_Params['AUC'][genre_i]),
                '7':'Precision:'+str(Test_Params['Precision'][genre_i]),
                '8':'Recall:'+str(Test_Params['Recall'][genre_i]),
                '9':'F1_score:'+str(Test_Params['F1_score'][genre_i]),
                '10':'AP (macro):'+str(Test_Params['average_precision']['macro']),
                '11':'AP (micro):'+str(Test_Params['average_precision']['micro']),
                '12':'AP (samples):'+str(Test_Params['average_precision']['samples']),
                '13':'AP (weighted):'+str(Test_Params['average_precision']['weighted']),
                }
            print_results(PARAMS, PARAMS['fold'],  **kwargs)
            all_fold_auc[genre_i].append(Test_Params['AUC'][genre_i])
            all_fold_prec[genre_i].append(Test_Params['Precision'][genre_i])
            all_fold_rec[genre_i].append(Test_Params['Recall'][genre_i])
            all_fold_fscore[genre_i].append(Test_Params['F1_score'][genre_i])
        all_fold_auc_micro_avg.append(Test_Params['AUC']['micro_avg'])
        all_fold_auc_macro_avg.append(Test_Params['AUC']['macro_avg'])
        all_fold_auc_macro_avg_weighted.append(Test_Params['AUC']['macro_avg_weighted'])
        all_fold_AP_macro.append(Test_Params['average_precision']['macro'])
        all_fold_AP_micro.append(Test_Params['average_precision']['micro'])
        all_fold_AP_samples.append(Test_Params['average_precision']['samples'])
        all_fold_AP_weighted.append(Test_Params['average_precision']['weighted'])

        Train_Params = None
        Test_Params = None
        del Train_Params, Test_Params
    
        if PARAMS['use_GPU']:
            reset_TF_session()
            ClearMemory()
        
    for genre_i in PARAMS['genre_list'].keys():
        if len(all_fold_auc_micro_avg)<PARAMS['CV_folds']:
            continue
        kwargs = {
            '0':'genre:'+genre_i,
            '1':'eval_loss:~',
            '2':'eval_auc:~',
            '3':'AUC (macro-avg):'+str(np.round(np.mean(all_fold_auc_macro_avg)*100,2))+'+-'+str(np.round(np.std(all_fold_auc_macro_avg)*100,2)),
            '4':'AUC (micro-avg):'+str(np.round(np.mean(all_fold_auc_micro_avg)*100,2))+'+-'+str(np.round(np.std(all_fold_auc_micro_avg)*100,2)),
            '5':'AUC (macro-avg weighted):'+str(np.round(np.mean(all_fold_auc_macro_avg_weighted)*100,2))+'+-'+str(np.round(np.std(all_fold_auc_macro_avg_weighted)*100,2)),
            '6':'AUC:'+str(np.round(np.mean(all_fold_auc[genre_i])*100,2))+'+-'+str(np.round(np.std(all_fold_auc[genre_i])*100,2)),
            '7':'Precision:'+str(np.round(np.mean(all_fold_prec[genre_i])*100,2))+'+-'+str(np.round(np.std(all_fold_prec[genre_i])*100,2)),
            '8':'Recall:'+str(np.round(np.mean(all_fold_rec[genre_i])*100,2))+'+-'+str(np.round(np.std(all_fold_rec[genre_i])*100,2)),
            '9':'F1_score:'+str(np.round(np.mean(all_fold_fscore[genre_i])*100,2))+'+-'+str(np.round(np.std(all_fold_fscore[genre_i])*100,2)),
            '10':'AP (macro):'+str(np.round(np.mean(all_fold_AP_macro)*100,2))+'+-'+str(np.round(np.std(all_fold_AP_macro)*100,2)),
            '11':'AP (micro):'+str(np.round(np.mean(all_fold_AP_micro)*100,2))+'+-'+str(np.round(np.std(all_fold_AP_micro)*100,2)),
            '12':'AP (samples):'+str(np.round(np.mean(all_fold_AP_samples)*100,2))+'+-'+str(np.round(np.std(all_fold_AP_samples)*100,2)),
            '13':'AP (weighted):'+str(np.round(np.mean(all_fold_AP_weighted)*100,2))+'+-'+str(np.round(np.std(all_fold_AP_weighted)*100,2)),
            }
        print_results(PARAMS, 'avg',  **kwargs)
        
    