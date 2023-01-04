#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 13:15:44 2021

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati

@reference: 
    A. Sharma, M. Jindal, A. Mittal, and D. K. Vishwakarma, “A unified audio 
    analysis framework for movie genre classification using movie trailers,” 
    in Proc. of the International Conference on Emerging Smart Computing and 
    Informatics (ESCI). IEEE, 2021, pp. 510–515.
    
Implementation details:
    1. 34-dimensional features from waveform:
        ZCR, energy, the entropy of energy, spectral centroid, spectral spread, 
        spectral entropy, spectral flux, spectral roll-off, 13-MFCC, 12-chroma,
        chroma deviation
    2. 34-dimensional features from differenced waveform:
        pyAudioAnalysis
    3. 50ms/25ms frame size and shift  is used
    4. Total of 200×68 feature matrix is obtained from 5s non-overlapping 
    chunks
    5. Every chunk is represented by a mean over 200 frames to obtain 1×68 
    dimensional feature vectors
    6. First and last chunk of every trailer is ignored
    7. The chunk representations are segmented using K-means clustering with 
    10 clusters using 400 trailers. The same trailers are not used for training
    and validation
    8. For every chunk, inverse of euclidian distance from each centroid is 
    computed
    9. The inverse distances are appended to the chunk features to form 
    78-dimension feature vector
    10. Mean over the feature vectors of all chunks in a trailer is computed 
    to obtain 1×78 dimension feature vector for every trailer
    11. The data is normalized in the range of 0 to 1
    12. Classified with AFA-net classifier
    13. Training/testing split is 85:15
    14. Learning rate of 0.001. Trained for 200 epochs
    15. Metric: AU(PRC)
    16. Genres: Action, Sci-Fi, Comedy, Horror, Romance
"""

import numpy as np
import os
import datetime
import lib.misc as misc
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Input, Dropout, Activation, BatchNormalization
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
import time
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
import sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from lib.pyAudioAnalysis import ShortTermFeatures as aF
from lib.pyAudioAnalysis import audioBasicIO as aIO 
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import librosa
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC





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



def reset_TF_session():
    tf.compat.v1.keras.backend.clear_session()




def AFA_net_model(output_dim):
    # create model
    input_layer = Input((78,))
    
    x = Dense(256, input_dim=(78,))(input_layer)
    x = BatchNormalization(axis=-1)(x)
    # x = Activation('relu')(x)

    x = Dense(256)(x)
    x = BatchNormalization(axis=-1)(x)
    # x = Activation('relu')(x)

    x = Dropout(0.4)(x)

    x = Dense(64)(x)
    x = BatchNormalization(axis=-1)(x)
    # x = Activation('relu')(x)

    x = Dense(32)(x)
    x = BatchNormalization(axis=-1)(x)
    # x = Activation('relu')(x)

    x = Dropout(0.2)(x)

    output_layer = Dense(output_dim, activation='sigmoid')(x)

    model = Model(input_layer, output_layer)
    
    learning_rate = 0.001
    opt = optimizers.Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[AUC(curve='PR')])

    return model, learning_rate




def train_model(PARAMS, data_dict, model, weightFile, logFile):
    csv_logger = CSVLogger(logFile, append=True)
    mcp = ModelCheckpoint(weightFile, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min', save_freq='epoch')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.000001, verbose=1, mode='min', min_delta=0.01)
    

    trainingTimeTaken = 0
    start = time.process_time()
    
    train_data = data_dict['train_data']
    train_label = data_dict['train_label']

    val_data = data_dict['val_data']
    val_label = data_dict['val_label']
    
    ''' Data augmentation real:synthetic::1:2 -- added 02-Jan-22'''
    # train_data_orig = train_data.copy()
    # train_label_orig = train_label.copy()
    # for num_augment in range(2):
    #     noisy_data = np.add(train_data_orig, np.random.normal(loc=0.0, scale=1e-5, size=np.shape(train_data_orig)))
    #     train_data = np.append(train_data, noisy_data, axis=0)
    #     noisy_labels = np.multiply(train_label_orig, np.array(np.random.rand(np.shape(train_label_orig)[0],np.shape(train_label_orig)[1])<0.95, dtype=int))
    #     train_label = np.append(train_label, noisy_labels, axis=0)
    # shuffle_idx = list(range(np.shape(train_data)[0]))
    # np.random.shuffle(shuffle_idx)
    # train_data = train_data[shuffle_idx,:]
    # train_label = train_label[shuffle_idx,:]    
    # print('train data: ', np.shape(train_data_orig), np.shape(train_data), np.shape(train_label))
    
    print('train data: ', np.shape(train_data), np.shape(train_label))
    print('genre_list: ', PARAMS['genre_list'])
        
    History = model.fit(
        x=train_data,
        y=train_label,
        epochs=PARAMS['epochs'],
        batch_size=PARAMS['batch_size'],
        verbose=1,
        validation_data=(val_data, val_label),
        callbacks=[csv_logger, mcp, reduce_lr],
        shuffle=True,
        # class_weight=PARAMS['genre_weights'],
        )
        
    trainingTimeTaken = time.process_time() - start
    print('Time taken for model training: ',trainingTimeTaken)

    return model, trainingTimeTaken, History




def perform_training(PARAMS, data_dict):
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
    
    input_dim = np.shape(data_dict['train_data'])[1]
    print('Weight file: ', weightFile, input_dim)
    if not os.path.exists(paramFile):
        model, learning_rate = AFA_net_model(len(PARAMS['genre_list']))
        misc.print_model_summary(summaryFile, model)
        print(model.summary())
        print('Architecture of Sharma et al., IEEE ESCI, 2021')
        
        model, trainingTimeTaken, History = train_model(PARAMS, data_dict, model, weightFile, logFile)
        if PARAMS['save_flag']:
            with open(architechtureFile, 'w') as f:
                f.write(model.to_json())
            np.savez(paramFile, lr=str(learning_rate), TTT=str(trainingTimeTaken))

    trainingTimeTaken = float(np.load(paramFile)['TTT'])
    # with open(architechtureFile, 'r') as f:
    #     model = model_from_json(f.read())
    model, learning_rate = AFA_net_model(len(PARAMS['genre_list']))
    model.load_weights(weightFile)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC(curve='PR')])

    print('Sharma et al. model exists! Loaded. Training time required=',trainingTimeTaken)
    # # print(model.summary())
    
    Train_Params = {
            'model': model,
            'trainingTimeTaken': trainingTimeTaken,
            'learning_rate': learning_rate,
            'paramFile': paramFile,
            'architechtureFile': architechtureFile,
            'weightFile': weightFile,
            }
    
    return Train_Params




def test_model(PARAMS, test_data, test_label, Train_Params):
    start = time.process_time()
    # loss, performance 
    loss, auc_measure = Train_Params['model'].evaluate(x=test_data, y=test_label)
    print('evaluation: ', loss, auc_measure)
    Predictions = Train_Params['model'].predict(test_data)
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
        # print(genre_i, ' pred: ', np.shape(pred))
        gt = test_label[:,PARAMS['genre_list'][genre_i]]

        precision_curve, recall_curve, threshold = precision_recall_curve(gt, pred)
        fscore_curve = np.divide(2*np.multiply(precision_curve, recall_curve), np.add(precision_curve, recall_curve)+1e-10)
        Precision[genre_i] = np.round(np.mean(precision_curve),4)
        Recall[genre_i] = np.round(np.mean(recall_curve),4)
        F1_score[genre_i] = np.round(np.mean(fscore_curve),4)
        P_curve[genre_i] = precision_curve
        R_curve[genre_i] = recall_curve
        AUC_values[genre_i] = np.round(auc(recall_curve, precision_curve)*100,2)
        # print(genre_i, ' AUC: ', AUC_values[genre_i])
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
    # fs, Xin = aIO.read_audio_file(PARAMS['audio_path']+'/'+fName.split('/')[-1])
    # Xin = aIO.stereo_to_mono(Xin)
    # print('Xin: ', np.shape(Xin))
    Xin, fs = librosa.load(PARAMS['audio_path']+'/'+fName.split('/')[-1], mono=True, sr=None)
    duration = np.round(len(Xin) / float(fs),2)
    print(fName, ' Xin: ', np.shape(Xin), ' fs=', fs, f'duration = {duration} seconds')
    
    Xin -= np.mean(Xin)
    Xin /= (np.max(Xin)-np.min(Xin))

    frameSize = int(PARAMS['Tw']*fs/1000) # frame size in samples
    frameShift = int(PARAMS['Ts']*fs/1000) # frame shift in samples
    chunk_size = 5*fs # 5s chunk size in samples
    # print(len(Xin), chunk_size)
    FV = np.empty([])
    for chunk_start in range(chunk_size,len(Xin),chunk_size): # Starting from 2nd chunk
        chunk_end = np.min([chunk_start+chunk_size, len(Xin)])
        if chunk_end==len(Xin): # ignoring last chunk, as described in paper
            continue
        Xin_chunk = Xin[chunk_start:chunk_end]
        [chunk_fv, feat_names] = aF.feature_extraction(Xin_chunk, fs, frameSize, frameShift)
        mean_chunk_fv = np.mean(chunk_fv, axis=1)
        # print('\tchunk_fv: ', np.shape(chunk_fv), np.shape(mean_chunk_fv))
        if np.size(FV)<=1:
            FV = np.array(mean_chunk_fv, ndmin=2)
        else:
            FV = np.append(FV, np.array(mean_chunk_fv, ndmin=2), axis=0)
    print('FV: ', np.shape(FV))
    return FV




def load_data_from_files(PARAMS, file_type):
    files = PARAMS[file_type+'_files'][PARAMS['classes'][0]]
    print('files: ', len(files))
    np.random.shuffle(files)
    genre_freq = {genre_i:0 for genre_i in PARAMS['genre_list'].keys()}

    if file_type=='train':
        kmeans_model_fName = PARAMS['opDir'] + '/kmeans_model_fold' + str(PARAMS['fold']) + '.pkl'
        if not os.path.exists(kmeans_model_fName):
            kmeans_data = np.empty([])
            kmeans_file_count = 0
            kmeans_files = []
            for fl in files:
                # print(fl, len(files))
                feat_fName = PARAMS['feature_opDir'] + '/' + fl.split('.')[0] + '.npy'
                if not fl.split('.')[0] in PARAMS['annotations'].keys():
                    print(feat_fName, ' not present in annotations')
                    continue
                if not os.path.exists(PARAMS['audio_path']+'/'+fl.split('/')[-1]):
                    print(feat_fName, ' wav file not present')
                    continue
        
                lab = np.zeros(len(PARAMS['genre_list']))
                # print(fl, np.shape(fv), np.shape(lab))
                for genre_i in PARAMS['genre_list'].keys():
                    if genre_i in PARAMS['annotations'][fl.split('.')[0]]['genre']:
                        lab[PARAMS['genre_list'][genre_i]] = 1
                        genre_freq[genre_i] += 1
                
                # print('lab: ', lab, PARAMS['annotations'][fl.split('.')[0]]['genre'])
                # To ignore trailers with no genre annotations
                if np.sum(lab)==0:
                    continue
                kmeans_files.append(fl) 
                
                # startTime = time.process_time()
                if not os.path.exists(feat_fName):
                    fv = compute_features(PARAMS, fl)
                    np.save(feat_fName, fv)
                    # print('feat computation time: ', time.process_time()-startTime)
                else:
                    fv = np.load(feat_fName)
                    # print('feat loading time: ', time.process_time()-startTime)
        
                fv[fv==-np.inf] = 0
                fv[fv==np.inf] = 0
                fv[fv==np.NAN] = 0
        
                if np.size(kmeans_data)<=1:
                    kmeans_data = fv
                else:
                    kmeans_data = np.append(kmeans_data, fv, axis=0)
                    
                kmeans_file_count += 1
                print(fl, kmeans_file_count, np.shape(kmeans_data))
                if kmeans_file_count==400:
                    break
            print('kmeans data: ', np.shape(kmeans_data))
            
            # files = np.setdiff1d(files, kmeans_files)
            # train_data = np.empty([])
            # for fl in files:
            #     feat_fName = PARAMS['feature_opDir'] + '/' + fl.split('.')[0] + '.npy'
            #     if not fl.split('.')[0] in PARAMS['annotations'].keys():
            #         print(feat_fName, ' not present in annotations')
            #         continue
            #     if not os.path.exists(PARAMS['audio_path']+'/'+fl.split('/')[-1]):
            #         print(feat_fName, ' wav file not present')
            #         continue    
            #     lab = np.zeros(len(PARAMS['genre_list']))
            #     # print(fl, np.shape(fv), np.shape(lab))
            #     for genre_i in PARAMS['genre_list'].keys():
            #         if genre_i in PARAMS['annotations'][fl.split('.')[0]]['genre']:
            #             lab[PARAMS['genre_list'][genre_i]] = 1
            #             genre_freq[genre_i] += 1
                
            #     # To ignore trailers with no genre annotations
            #     if np.sum(lab)==0:
            #         continue
                
            #     # startTime = time.process_time()
            #     if not os.path.exists(feat_fName):
            #         fv = compute_features(PARAMS, fl)
            #         np.save(feat_fName, fv)
            #         # print('feat computation time: ', time.process_time()-startTime)
            #     else:
            #         fv = np.load(feat_fName)
            #         # print('feat loading time: ', time.process_time()-startTime)
        
            #     fv[fv==-np.inf] = 0
            #     fv[fv==np.inf] = 0
            #     fv[fv==np.NAN] = 0
        
            #     if np.size(train_data)<=1:
            #         train_data = fv
            #     else:
            #         train_data = np.append(train_data, fv, axis=0)
            # print('train data: ', np.shape(train_data))    
            
            # all_data = np.append(train_data, kmeans_data, axis=0)
            # std_scaler = StandardScaler().fit(all_data)
            # kmeans_data = std_scaler.transform(kmeans_data)
            # # minmax_scaler = MinMaxScaler().fit(all_data)
            # # kmeans_data = minmax_scaler.transform(kmeans_data)
            
            kmeans_model = KMeans(n_clusters=PARAMS['kmeans_nClusters'], verbose=1)
            kmeans_model.fit(kmeans_data)
            # misc.save_obj({'kmeans_model':kmeans_model, 'std_scaler':std_scaler, 'train_files':files, 'genre_freq':genre_freq}, PARAMS['opDir'], 'kmeans_model_fold' + str(PARAMS['fold']))
            # misc.save_obj({'kmeans_model':kmeans_model, 'minmax_scaler':minmax_scaler, 'train_files':files, 'genre_freq':genre_freq}, PARAMS['opDir'], 'kmeans_model_fold' + str(PARAMS['fold']))
            misc.save_obj({'kmeans_model':kmeans_model, 'train_files':files, 'genre_freq':genre_freq}, PARAMS['opDir'], 'kmeans_model_fold' + str(PARAMS['fold']))

    kmeans_model = misc.load_obj(PARAMS['opDir'], 'kmeans_model_fold' + str(PARAMS['fold']))['kmeans_model']
    # std_scaler = misc.load_obj(PARAMS['opDir'], 'kmeans_model_fold' + str(PARAMS['fold']))['std_scaler']
    # minmax_scaler = misc.load_obj(PARAMS['opDir'], 'kmeans_model_fold' + str(PARAMS['fold']))['minmax_scaler']
    train_files = misc.load_obj(PARAMS['opDir'], 'kmeans_model_fold' + str(PARAMS['fold']))['train_files']
    genre_freq = misc.load_obj(PARAMS['opDir'], 'kmeans_model_fold' + str(PARAMS['fold']))['genre_freq']
    
    centroids = kmeans_model.cluster_centers_
    print('centroids: ', np.shape(centroids))
    data = np.empty([])
    label = np.empty([])
    for fl in files:
        feat_fName = PARAMS['feature_opDir'] + '/' + fl.split('.')[0] + '.npy'
        if not fl.split('.')[0] in PARAMS['annotations'].keys():
            print(feat_fName, ' not present in annotations')
            continue
        if not os.path.exists(PARAMS['audio_path']+'/'+fl.split('/')[-1]):
            print(feat_fName, ' wav file not present')
            continue
        if not os.path.exists(feat_fName):
            fv = compute_features(PARAMS, fl)
            np.save(feat_fName, fv)
        else:
            fv = np.load(feat_fName)
        fv[fv==-np.inf] = 0
        fv[fv==np.inf] = 0
        fv[fv==np.NAN] = 0
        
        # fv = std_scaler.transform(fv)
        # fv = minmax_scaler.transform(fv)
        
        dist = pairwise_distances(fv, centroids)
        # print(np.round(dist,2))
        # print(centroids)
        inv_dist = np.divide(1,dist+1e-10)
        fv = np.append(fv, inv_dist, axis=1)
        lab = np.zeros(len(PARAMS['genre_list']))
        # print(fl, np.shape(fv), np.shape(lab))
        for genre_i in PARAMS['genre_list'].keys():
            if genre_i in PARAMS['annotations'][fl.split('.')[0]]['genre']:
                lab[PARAMS['genre_list'][genre_i]] = 1
                genre_freq[genre_i] += 1
        
        # To ignore trailers with no genre annotations
        if np.sum(lab)==0:
            continue
        
        if np.size(data)<=1:
            data = np.array(np.mean(fv, axis=0), ndmin=2)
            label = np.array(lab, ndmin=2)
        else:
            data = np.append(data, np.array(np.mean(fv, axis=0), ndmin=2), axis=0)
            label = np.append(label, np.array(lab, ndmin=2), axis=0)
    print(file_type, 'data: ', np.shape(data), np.shape(label))

    files = train_files
    for key in genre_freq.keys():
        genre_freq[key] /= len(PARAMS[file_type+'_files'][PARAMS['classes'][0]])
    
    return data, label, genre_freq




def get_data(PARAMS):
    train_data, train_label, genre_freq_train = load_data_from_files(PARAMS, file_type='train')
    std_scaler = StandardScaler().fit(train_data)
    train_data = std_scaler.transform(train_data)
    # min_max_scaler = MinMaxScaler().fit(train_data)
    # train_data = min_max_scaler.transform(train_data)
    print('train data loaded: ', np.shape(train_data), np.shape(train_label))

    val_data, val_label, genre_freq_val = load_data_from_files(PARAMS, file_type='val')
    val_data = std_scaler.transform(val_data)
    # val_data = min_max_scaler.transform(val_data)
    print('val data loaded: ', np.shape(val_data), np.shape(val_label))
    
    test_data, test_label, genre_freq_test = load_data_from_files(PARAMS, file_type='test')
    test_data = std_scaler.transform(test_data)
    # test_data = min_max_scaler.transform(test_data)
    print('test data loaded: ', np.shape(test_data), np.shape(test_label))
    
    data_dict = {}
    data_dict['std_scaler'] = std_scaler
    # data_dict['min_max_scaler'] = min_max_scaler
    data_dict['train_data'] = train_data
    data_dict['train_label'] = train_label
    data_dict['val_data'] = val_data
    data_dict['val_label'] = val_label
    data_dict['test_data'] = test_data
    data_dict['test_label'] = test_label
    
    return data_dict, genre_freq_train




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
            # 'audio_path': '/media/mrinmoy/Windows_Volume/PhD_Work/data/movie-trailers-dataset-master/wav/',
            # 'annot_path': '/media/mrinmoy/Windows_Volume/PhD_Work/data/movie-trailers-dataset-master/Annotations_6genre.csv',

            'CV_folds': 3,
            'fold': 0,
            'save_flag': True,
            'use_GPU': True,
            'GPU_session':None,
            'classes':{0:'wav'},
            'epochs': 200,
            'batch_size': 16,
            'W': 5000, # interval size in ms
            'W_shift': 5000, # interval shift in ms
            'Tw': 50, # frame size in ms
            'Ts': 25, # frame shift in ms
            'kmeans_nClusters': 10,
            # 'possible_genres': ['Action', 'Fantasy', 'Sci-Fi', 'Thriller', 'Romance', 'Family', 'Mystery', 'Comedy', 'Drama', 'Animation', 'Crime', 'Horror', 'Biography', 'Adventure', 'Music', 'War', 'History', 'Sport', 'Musical', 'Documentary', 'Western', 'Film-Noir', 'Short', 'News', 'Reality-TV', 'Game-Show'], # Moviescope all
            # 'possible_genres': ['Action', 'Sci-Fi', 'Romance', 'Comedy', 'Horror'], # Sharma et al. 5
            # 'possible_genres': ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Thriller'], # 6 genres EmoGDB
            'possible_genres': ['Action', 'Animation', 'Biography', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller'], # Moviescope 13
            }

    PARAMS['CV_path_train'] = './cross_validation_info/' + PARAMS['dataset_name'] + '/'    
    if not os.path.exists(PARAMS['CV_path_train']+'/cv_file_list_original.pkl'):
        print('Fold division of files for cross-validation not done!')
        sys.exit(0)
    PARAMS['cv_file_list'] = misc.load_obj(PARAMS['CV_path_train'], 'cv_file_list_original')
    
    n_classes = len(PARAMS['classes'])
    DT_SZ = 0
    for clNum in PARAMS['classes'].keys():
        classname = PARAMS['classes'][clNum]
        DT_SZ += PARAMS['cv_file_list']['total_duration'][classname] # in Hours
    DT_SZ *= 3600*1000 # in msec
    # tr_frac = ((PARAMS['CV_folds']-1)/PARAMS['CV_folds'])*0.7
    # vl_frac = ((PARAMS['CV_folds']-1)/PARAMS['CV_folds'])*0.3
    # ts_frac = (1/PARAMS['CV_folds'])
    # PARAMS['TR_STEPS'] = int(np.floor(DT_SZ/PARAMS['W_shift'])*tr_frac/(n_classes*PARAMS['batch_size']))
    # PARAMS['V_STEPS'] = int(np.floor(DT_SZ/PARAMS['W_shift'])*vl_frac/(n_classes*PARAMS['batch_size']))
    # PARAMS['TS_STEPS'] = int(np.floor(DT_SZ/PARAMS['W_shift'])*ts_frac/(n_classes*PARAMS['batch_size']))
    # print('TR_STEPS: %d, \tV_STEPS: %d, \tTS_STEPS: %d\n'%(PARAMS['TR_STEPS'], PARAMS['V_STEPS'], PARAMS['TS_STEPS']))

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
    
    PARAMS['feature_opDir'] = PARAMS['feature_path'] + '/' + PARAMS['dataset_name'] + '/Sharma_et_al_ECSI_2021/'
    if not os.path.exists(PARAMS['feature_opDir']):
        os.makedirs(PARAMS['feature_opDir'])

    PARAMS['opDir'] = './results/' + PARAMS['dataset_name'] + '/' + PARAMS['today'] + '/Sharma_et_al_ECSI_2021/'
    if not os.path.exists(PARAMS['opDir']):
        os.makedirs(PARAMS['opDir'])

    PARAMS['annotations'], PARAMS['genre_list'] = misc.get_annotations(PARAMS['annot_path'], PARAMS['possible_genres'])
    print('genre_list: ', PARAMS['genre_list'])

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

    # for PARAMS['fold'] in range(PARAMS['CV_folds']):
    #     PARAMS['train_files'], PARAMS['test_files'] = get_train_test_files(PARAMS)
    #     print('train_files: ', PARAMS['train_files'])
    #     print('test_files: ', PARAMS['test_files'])

    for PARAMS['fold'] in range(1): # Original split
        PARAMS['train_files'] = {'wav':PARAMS['cv_file_list']['original_splits']['train']}
        PARAMS['val_files'] = {'wav':PARAMS['cv_file_list']['original_splits']['val']}
        PARAMS['test_files'] = {'wav':PARAMS['cv_file_list']['original_splits']['test']}
        print('train_files: ', PARAMS['train_files'])
        print('test_files: ', PARAMS['test_files'])

        if PARAMS['use_GPU']:
            PARAMS['GPU_session'] = start_GPU_session()
        
        if not os.path.exists(PARAMS['opDir']+'/data_dict_fold' + str(PARAMS['fold']) + '.pkl'):
            data_dict, PARAMS['genre_freq'] = get_data(PARAMS)
            misc.save_obj({'data_dict':data_dict, 'genre_freq':PARAMS['genre_freq']}, PARAMS['opDir'], 'data_dict_fold' + str(PARAMS['fold']))
        else:
            data_dict = misc.load_obj(PARAMS['opDir'], 'data_dict_fold' + str(PARAMS['fold']))['data_dict']
            PARAMS['genre_freq'] = misc.load_obj(PARAMS['opDir'], 'data_dict_fold' + str(PARAMS['fold']))['genre_freq']
        print('Train data loaded: ', np.shape(data_dict['train_data']), np.shape(data_dict['train_label']))
        print('Val data loaded: ', np.shape(data_dict['val_data']), np.shape(data_dict['val_label']))
        print('Test data loaded: ', np.shape(data_dict['test_data']), np.shape(data_dict['test_label']))

        PARAMS['genre_weights'] = {}
        max_freq = np.max([PARAMS['genre_freq'][genre_i] for genre_i in PARAMS['genre_freq'].keys()])
        for genre_i in PARAMS['genre_freq'].keys():
            lab = PARAMS['genre_list'][genre_i]
            PARAMS['genre_weights'][lab] = max_freq/PARAMS['genre_freq'][genre_i]
        
        # import sys
        # sys.exit(0)
        # continue
            
        PARAMS['modelName'] = PARAMS['opDir'] + '/fold' + str(PARAMS['fold']) + '_model.xyz'
        Train_Params = perform_training(PARAMS, data_dict)

        if not os.path.exists(PARAMS['opDir']+'/Test_Params_fold'+str(PARAMS['fold'])+'.pkl'):
            Test_Params = test_model(PARAMS, data_dict['test_data'], data_dict['test_label'], Train_Params)
            misc.save_obj(Test_Params, PARAMS['opDir'], 'Test_Params_fold'+str(PARAMS['fold']))
        else:
            Test_Params = misc.load_obj(PARAMS['opDir'], 'Test_Params_fold'+str(PARAMS['fold']))
        
        # print('Test keys: ', Test_Params['AUC'].keys())
        for genre_i in PARAMS['genre_list'].keys():
            # print(genre_i, 'AUC=', Test_Params['AUC'][genre_i])
            kwargs = {
                '0':'genre:'+genre_i,
                '1':'AUC (micro-avg):'+str(Test_Params['AUC']['micro_avg']),
                '2':'AUC (macro-avg):'+str(Test_Params['AUC']['macro_avg']),
                '3':'AUC (macro-avg weighted):'+str(Test_Params['AUC']['macro_avg_weighted']),
                '4':'AUC:'+str(Test_Params['AUC'][genre_i]),
                '5':'Precision:'+str(Test_Params['Precision'][genre_i]),
                '6':'Recall:'+str(Test_Params['Recall'][genre_i]),
                '7':'F1_score:'+str(Test_Params['F1_score'][genre_i]),
                '8':'AP (macro):'+str(Test_Params['average_precision']['macro']),
                '9':'AP (micro):'+str(Test_Params['average_precision']['micro']),
                '10':'AP (samples):'+str(Test_Params['average_precision']['samples']),
                '11':'AP (weighted):'+str(Test_Params['average_precision']['weighted']),
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
        
    for genre_i in PARAMS['genre_list'].keys():
        # print(genre_i, 'AUC=', all_fold_auc_micro_avg, len(all_fold_auc_micro_avg))
        if len(all_fold_auc_micro_avg)<PARAMS['CV_folds']:
            continue
        kwargs = {
            '0':'genre:'+genre_i,
            '1':'AUC (micro-avg):'+str(np.round(np.mean(all_fold_auc_micro_avg)*100,2))+'+-'+str(np.round(np.std(all_fold_auc_micro_avg)*100,2)),
            '2':'AUC (macro-avg):'+str(np.round(np.mean(all_fold_auc_macro_avg)*100,2))+'+-'+str(np.round(np.std(all_fold_auc_macro_avg)*100,2)),
            '3':'AUC (macro-avg weighted):'+str(np.round(np.mean(all_fold_auc_macro_avg_weighted)*100,2))+'+-'+str(np.round(np.std(all_fold_auc_macro_avg_weighted)*100,2)),
            '4':'AUC:'+str(np.round(np.mean(all_fold_auc[genre_i])*100,2))+'+-'+str(np.round(np.std(all_fold_auc[genre_i])*100,2)),
            '5':'Precision:'+str(np.round(np.mean(all_fold_prec[genre_i])*100,2))+'+-'+str(np.round(np.std(all_fold_prec[genre_i])*100,2)),
            '6':'Recall:'+str(np.round(np.mean(all_fold_rec[genre_i])*100,2))+'+-'+str(np.round(np.std(all_fold_rec[genre_i])*100,2)),
            '7':'F1_score:'+str(np.round(np.mean(all_fold_fscore[genre_i])*100,2))+'+-'+str(np.round(np.std(all_fold_fscore[genre_i])*100,2)),
            '8':'AP (macro):'+str(np.round(np.mean(all_fold_AP_macro)*100,2))+'+-'+str(np.round(np.std(all_fold_AP_macro)*100,2)),
            '9':'AP (micro):'+str(np.round(np.mean(all_fold_AP_micro)*100,2))+'+-'+str(np.round(np.std(all_fold_AP_micro)*100,2)),
            '10':'AP (sample):'+str(np.round(np.mean(all_fold_AP_samples)*100,2))+'+-'+str(np.round(np.std(all_fold_AP_samples)*100,2)),
            '11':'AP (weighted):'+str(np.round(np.mean(all_fold_AP_weighted)*100,2))+'+-'+str(np.round(np.std(all_fold_AP_weighted)*100,2)),
            }
        print_results(PARAMS, 'avg',  **kwargs)
        
    