#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 22:43:34 2022

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import os
import datetime
import lib.misc as misc
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json
# from tensorflow.keras.initializers import he_uniform
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt





def load_trained_model(PARAMS):
    modelName = '@'.join(PARAMS['modelName'].split('.')[:-1]) + '.' + PARAMS['modelName'].split('.')[-1]
    weightFile = modelName.split('.')[0] + '.h5'
    architechtureFile = modelName.split('.')[0] + '.json'
    paramFile = modelName.split('.')[0] + '_params.npz'
    logFile = modelName.split('.')[0] + '_log.csv'

    modelName = '.'.join(modelName.split('@'))
    weightFile = '.'.join(weightFile.split('@'))
    architechtureFile = '.'.join(architechtureFile.split('@'))
    paramFile = '.'.join(paramFile.split('@'))
    logFile = '.'.join(logFile.split('@'))
    
    output_dim = len(PARAMS['classes'])
    print(output_dim)

    print('Weight file: ', weightFile)
    if os.path.exists(paramFile):
        learning_rate = float(np.load(paramFile)['lr'])
        trainingTimeTaken = float(np.load(paramFile)['TTT'])
        optimizerName = 'Adam'

        # Model reconstruction from JSON file
        with open(architechtureFile, 'r') as f:
            model = model_from_json(f.read())
        # Load weights into the new model
        model.load_weights(weightFile)
        opt = optimizers.Adam(lr=learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        std_scaler = misc.load_obj(PARAMS['model_folder'], 'std_scaler_fold'+str(PARAMS['SM_classifier_fold']))

        print('DNN model exists! Loaded. Training time required=',trainingTimeTaken)
        # print(model.summary())
    
        Train_Params = {
            'model': model,
            'trainingTimeTaken': trainingTimeTaken,
            'learning_rate': learning_rate,
            'optimizerName': optimizerName,
            'paramFile': paramFile,
            'architechtureFile': architechtureFile,
            'weightFile': weightFile,
            'std_scaler': std_scaler,
            }
    else:
        Train_Params = {}
    
    return Train_Params




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




def get_train_test_files(cv_file_list, numCV, foldNum):
    train_files = {}
    test_files = {}
    classes = {'music':0, 'speech':1}
    for class_name in classes.keys():
        train_files[class_name] = []
        test_files[class_name] = []
        for i in range(numCV):
            files = cv_file_list[class_name]['fold'+str(i)]
            if foldNum==i:
                test_files[class_name].extend(files)
            else:
                train_files[class_name].extend(files)
    
    return train_files, test_files




def __init__():
    PARAMS = {
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),

            # Laptop
            # 'dataset_name': 'EmoGDB',
            # 'path': './features/EmoGDB/1s_100ms_10ms_5ms_10PT/',
            # 'model_folder': './results/musan/CBoW-ASPT-LSPT_mix5/',
            # 'annot_path': '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/EmoGDB/Annotations_6genre.csv',

            # LabPC
            # 'dataset_name': 'movie-trailers-dataset-master',
            # 'path': '/home/mrinmoy/Documents/PhD_Work/Features/MTGC_SMO/Moviescope/1s_1s_10ms_5ms_10PT_2021-12-22/',
            # 'model_folder': './results/musan/CBoW-ASPT-LSPT_mix5/',
            # 'annot_path': '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/movie-trailers-dataset-master/Annotations.csv',

            # LabPC
            # 'dataset_name': 'Moviescope',
            # 'path': '/home/mrinmoy/Documents/PhD_Work/Features/MTGC_SMO/Moviescope/1s_1s_10ms_5ms_10PT_2022-01-21/',
            # 'model_folder': './results/musan/CBoW-ASPT-LSPT_mix5/',
            # 'annot_path': '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/Moviescope/Original_Metadata.csv',

            # EEE-GPU
            'dataset_name': 'Moviescope',
            'path': '/home1/PhD/mrinmoy.bhattacharjee/MTGC_SMO/Moviescope/X-Vectors/',
            'model_folder': './results/musan/X-Vectors/',
            'annot_path': '/home1/PhD/mrinmoy.bhattacharjee/data/Moviescope/Original_Metadata.csv',

            'featName': 'X-Vectors', # 'CBoW-ASPT-LSPT',
            'SM_classifier_fold': 0,
            'CV_folds': 3,
            'fold': 0,
            'save_flag': True,
            'use_GPU': True,
            'GPU_session':None,
            'classes':{0:'music', 1:'speech'},
            'high_confidence_thresh': 0.85,
            'plot_fig': True,
            # 'possible_genres': ['Action', 'Fantasy', 'Sci-Fi', 'Thriller', 'Romance', 'Family', 'Mystery', 'Comedy', 'Drama', 'Animation', 'Crime', 'Horror', 'Biography', 'Adventure', 'Music', 'War', 'History', 'Sport', 'Musical', 'Documentary', 'Western', 'Film-Noir', 'Short', 'News', 'Reality-TV', 'Game-Show'], # Moviescope all
            'possible_genres': ['Action', 'Animation', 'Biography', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller'], # Moviescope 13
            }

    return PARAMS



if __name__ == '__main__':
    PARAMS = __init__()

    PARAMS['opDir'] = PARAMS['path'] + '/SpMu_Predictions_' + PARAMS['featName'] + '/'
    if not os.path.exists(PARAMS['opDir']):
        os.makedirs(PARAMS['opDir'])
        
    PARAMS['feature_folder'] = PARAMS['path'] + '/wav/'

    if PARAMS['use_GPU']:
        PARAMS['GPU_session'] = start_GPU_session()

    ''' Set training parameters '''
    PARAMS['modelName'] = PARAMS['model_folder'] + '/fold' + str(PARAMS['SM_classifier_fold']) + '_model.xyz'

    Train_Params = load_trained_model(PARAMS)
    SM_Pred = {}
    files = librosa.util.find_files(PARAMS['feature_folder'], ext=['npy'])
    for fl in files:
        if not os.path.exists(fl):
            print(f'{fl} does not exists')
            continue
        opt_fName = PARAMS['opDir'] + '/' + fl.split('/')[-1].split('.')[0] + '.npy'
        if not os.path.exists(opt_fName):
            fv = np.load(fl)
            fv = Train_Params['std_scaler'].transform(fv)
            pred = Train_Params['model'].predict(fv)
            np.save(opt_fName, pred)
            print(fl.split('/')[-1], np.shape(fv))
        else:
            pred = np.load(opt_fName)
        print(fl.split('/')[-1], np.shape(pred), np.shape(np.array(pred[:,1], ndmin=2)))
        SM_Pred[fl.split('/')[-1].split('.')[0]] = np.array(pred[:,1], ndmin=2)
    
    annotations, genre_list = misc.get_annotations(PARAMS['annot_path'], PARAMS['possible_genres'])
    genre_sp_mu_dist = {}
    audio_type_total = [0,0,0]
    for annot_i in SM_Pred.keys():
        if not os.path.exists(PARAMS['feature_folder']+'/'+annot_i+'.npy'):
            continue
        print(f'SM_Pred={SM_Pred[annot_i].shape}')
        pred = SM_Pred[annot_i].flatten()
        sp_seg = np.sum(pred>=PARAMS['high_confidence_thresh'])
        mu_seg = np.sum(pred<=(1-PARAMS['high_confidence_thresh']))
        ot_seg = len(pred)-sp_seg-mu_seg
        print(f'{annot_i} {[mu_seg, sp_seg, ot_seg]}')
        genres = annotations[annot_i]['genre']
        print(f'genres={genres}')
        # for genre_i in genre_list.keys():
        for genre_i in genres:
            if genre_i not in genre_sp_mu_dist.keys():
                genre_sp_mu_dist[genre_i] = [mu_seg, sp_seg, ot_seg]
            else:
                genre_sp_mu_dist[genre_i] = np.add(genre_sp_mu_dist[genre_i], [mu_seg, sp_seg, ot_seg])
        audio_type_total = np.add(audio_type_total, [mu_seg, sp_seg, ot_seg])
    
    print(f'audio_type_total={audio_type_total}')
    for genre_i in genre_list.keys():
        print(f'{genre_i} {genre_sp_mu_dist[genre_i]}')
    if PARAMS['use_GPU']:
        reset_TF_session()
        
    print('genre_list: ', genre_list)
    # print('annotations: ', annotations)

    if PARAMS['plot_fig']:
        gen_hist = np.zeros((3,len(genre_list)))
        for genre_i in genre_list.keys():
            gen_hist[0,genre_list[genre_i]] = genre_sp_mu_dist[genre_i][0]
            gen_hist[1,genre_list[genre_i]] = genre_sp_mu_dist[genre_i][1]
            gen_hist[2,genre_list[genre_i]] = genre_sp_mu_dist[genre_i][2]
        print('gen_hist: ', np.shape(gen_hist), np.shape(np.sum(gen_hist, axis=1)))
        gen_hist = np.divide(gen_hist, np.repeat(np.array(np.sum(gen_hist, axis=1), ndmin=2).T, np.shape(gen_hist)[1], axis=1))
        gen_hist = np.divide(gen_hist, np.repeat(np.array(np.max(gen_hist, axis=0), ndmin=2), np.shape(gen_hist)[0], axis=0))
        print(f'gen_hist={gen_hist}')

        plt_count = 1
        for genre_i in genre_list.keys():
            plt.subplot(5,3,plt_count)
            plt.bar(['Music', 'Speech', 'Others'], gen_hist[:,genre_list[genre_i]], color='blue', width=0.8)
            plt.ylim([0.7,np.max(gen_hist)])
            plt.title(genre_i)
            plt_count += 1
        plt.show()
        plt.savefig(PARAMS['path']+'/Genre_Wise_thresh'+str(int(PARAMS['high_confidence_thresh']*100))+'.png', bbox_inches='tight')
        print(PARAMS['path']+'/Genre_Wise_thresh'+str(int(PARAMS['high_confidence_thresh']*100))+'.png')
        
        plt.figure()
        aud_hist = np.zeros((3,len(genre_list)))
        for genre_i in genre_sp_mu_dist.keys():
            aud_hist[0,genre_list[genre_i]] = genre_sp_mu_dist[genre_i][0]
            aud_hist[1,genre_list[genre_i]] = genre_sp_mu_dist[genre_i][1]
            aud_hist[2,genre_list[genre_i]] = genre_sp_mu_dist[genre_i][2]
        aud_hist = np.divide(aud_hist, np.repeat(np.array(np.sum(aud_hist, axis=0), ndmin=2), np.shape(aud_hist)[0], axis=0))
        aud_hist = np.divide(aud_hist, np.repeat(np.array(np.max(aud_hist, axis=1), ndmin=2).T, np.shape(aud_hist)[1], axis=1))
        
        plt.subplot(311)
        plt.bar(genre_sp_mu_dist.keys(), aud_hist[0,:], color='blue', width=0.4)
        plt.ylim([0.7,np.max(aud_hist)])
        plt.title('Music')
    
        plt.subplot(312)
        plt.bar(genre_sp_mu_dist.keys(), aud_hist[1,:], color='maroon', width=0.4)
        plt.ylim([0.7,np.max(aud_hist)])
        plt.title('Speech')
    
        plt.subplot(313)
        plt.bar(genre_sp_mu_dist.keys(), aud_hist[2,:], color='green', width=0.4)
        plt.ylim([0.7,np.max(aud_hist)])
        plt.title('Others')
        plt.show()
        plt.savefig(PARAMS['path']+'/Audio_Wise_thresh'+str(int(PARAMS['high_confidence_thresh']*100))+'.png', bbox_inches='tight')
        print(PARAMS['path']+'/Audio_Wise_thresh'+str(int(PARAMS['high_confidence_thresh']*100))+'.png')
        