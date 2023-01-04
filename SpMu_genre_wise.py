#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 01:48:27 2022

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import os
import datetime
import lib.misc as misc
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import librosa
import csv
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from scipy.io import savemat




def __init__():
    PARAMS = {
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),

            # Laptop
            # 'dataset_name': 'EmoGDB',
            # 'path': './features/EmoGDB/1s_100ms_10ms_5ms_10PT/',
            # 'model_folder': './results/musan/CBoW-ASPT-LSPT_mix5/',
            # 'annot_path': '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/EmoGDB/Annotations_6genre.csv',

            # Laptop
            'dataset_name': 'Moviescope',
            'path': './features/Moviescope/1s_1s_10ms_5ms_10PT_2022-01-21/',
            'audio_path': '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/Moviescope/wav/',
            'annot_path': '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/Moviescope/Original_Metadata.csv',

            # LabPC
            # 'dataset_name': 'movie-trailers-dataset-master',
            # 'path': '/home/mrinmoy/Documents/PhD_Work/Features/MTGC_SMO/Moviescope/1s_1s_10ms_5ms_10PT_2021-12-22/',
            # 'model_folder': './results/musan/CBoW-ASPT-LSPT_mix5/',
            # 'annot_path': '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/movie-trailers-dataset-master/Annotations.csv',

            # LabPC
            # 'dataset_name': 'Moviescope',
            # 'path': '/home/mrinmoy/Documents/PhD_Work/Features/MTGC_SMO/Moviescope/1s_1s_10ms_5ms_10PT_2022-01-25/',
            # 'model_folder': './results/musan/CBoW-ASPT-LSPT_mix5/',
            # 'annot_path': '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/Moviescope/Annotations_13genres.csv',

            'featName': 'CBoW-ASPT-LSPT',
            'SM_classifier_fold': 0,
            'CV_folds': 3,
            'fold': 0,
            'save_flag': True,
            'use_GPU': False,
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

    PARAMS['opDir'] = PARAMS['path'] + '/Details/'
    if not os.path.exists(PARAMS['opDir']):
        os.makedirs(PARAMS['opDir'])
    
    PARAMS['feature_path'] = PARAMS['path'] + '/SpMu_Predictions_CBoW-ASPT-LSPT/'
        
    SM_Pred = {}
    files = librosa.util.find_files(PARAMS['feature_path'], ext=['npy'])
    All_Pred = np.empty([])
    for fl in files:
        opt_fName = PARAMS['feature_path'] + '/' + fl.split('/')[-1].split('.')[0] + '.npy'
        pred = np.load(opt_fName)
        print(fl.split('/')[-1], np.shape(pred), np.shape(np.array(pred[:,1], ndmin=2)))
        SM_Pred[fl.split('/')[-1].split('.')[0]] = np.array(pred[:,1], ndmin=2)
        if np.size(All_Pred)<=1:
            All_Pred = pred
        else:
            All_Pred = np.append(All_Pred, pred, axis=0)
    scaler = MinMaxScaler().fit(All_Pred)
    
    annotations, genre_list = misc.get_annotations(PARAMS['annot_path'], PARAMS['possible_genres'])
    genre_freq = {genre_i:0 for genre_i in genre_list.keys()}
    tot_segments = 0
    genre_sp_mu_dist = {}
    audio_type_total = [0,0,0]
    speech_music = {}
    spmu_thresh = 0.2
    for annot_i in annotations.keys():
        if not os.path.exists(PARAMS['feature_path']+'/'+annot_i+'.npy'):
            print(PARAMS['feature_path']+'/'+annot_i+'.npy')
            continue
        if annot_i not in SM_Pred.keys():
            print(annot_i)
            continue
        for genre_i in genre_list.keys():
            if genre_i in annotations[annot_i.split('.')[0]]['genre']:
                genre_freq[genre_i] += 1
                tot_segments += 1

        pred = SM_Pred[annot_i].flatten()
        pred = savgol_filter(pred, 11, polyorder=2)
        sp_seg = np.sum(pred>=PARAMS['high_confidence_thresh'])
        mu_seg = np.sum(pred<=(1-PARAMS['high_confidence_thresh']))
        ot_seg = len(pred)-sp_seg-mu_seg
        for genre_i in genre_list.keys():
            if not genre_i in genre_sp_mu_dist:
                genre_sp_mu_dist[genre_i] = [mu_seg, sp_seg, ot_seg]
            else:
                genre_sp_mu_dist[genre_i] = np.add(genre_sp_mu_dist[genre_i], [mu_seg, sp_seg, ot_seg])
            audio_type_total = np.add(audio_type_total, [mu_seg, sp_seg, ot_seg])
            
        for genre_j in annotations[annot_i]['genre']:
            pred_scaled = np.append(1-np.array(pred, ndmin=2).T, np.array(pred, ndmin=2).T, axis=1)
            pred_scaled = scaler.transform(pred_scaled)
            if genre_j in speech_music.keys():
                # speech_music[genre_j][0] += np.sum(pred<=spmu_thresh)
                # speech_music[genre_j][1] += np.sum(pred>spmu_thresh)
                speech_music[genre_j][0] += np.sum(pred_scaled[:,0])
                speech_music[genre_j][1] += np.sum(pred_scaled[:,1])
            else:
                # speech_music[genre_j] = {0:np.sum(pred<=spmu_thresh), 1:np.sum(pred>spmu_thresh)}
                speech_music[genre_j] = {0:np.sum(pred_scaled[:,0]), 1:np.sum(pred_scaled[:,1])}
    
    # min_freq = 1e10
    # for genre_i in genre_list.keys():
    #     genre_freq[genre_i] /= tot_segments
    #     if genre_freq[genre_i]<min_freq:
    #         min_freq = genre_freq[genre_i]
    # for genre_i in genre_list.keys():
    #     genre_weight = min_freq/(genre_freq[genre_i]+1e-10)
    #     speech_music[genre_j][0] *= genre_weight
    #     speech_music[genre_j][1] *= genre_weight
                
    print('genre_list: ', genre_list)
    print('genre_sp_mu_dist: ', genre_sp_mu_dist.keys())
    # print('annotations: ', annotations)

    if PARAMS['plot_fig']:
        gen_hist = np.zeros((3,len(genre_list)))
        for genre_i in genre_list.keys():
            gen_hist[0,genre_list[genre_i]] = genre_sp_mu_dist[genre_i][0]
            gen_hist[1,genre_list[genre_i]] = genre_sp_mu_dist[genre_i][1]
            gen_hist[2,genre_list[genre_i]] = genre_sp_mu_dist[genre_i][2]
        print('gen_hist: ', np.shape(gen_hist), np.shape(np.sum(gen_hist, axis=1)))
        # gen_hist = np.divide(gen_hist, np.repeat(np.array(np.sum(gen_hist, axis=0), ndmin=2), np.shape(gen_hist)[0], axis=0))
        gen_hist = np.divide(gen_hist, np.repeat(np.array(np.sum(gen_hist, axis=1), ndmin=2).T, np.shape(gen_hist)[1], axis=1))
        gen_hist = np.divide(gen_hist, np.repeat(np.array(np.max(gen_hist, axis=0), ndmin=2), np.shape(gen_hist)[0], axis=0))

        plt_count = 1
        for genre_i in genre_list.keys():
            plt.subplot(5,3,plt_count)
            plt.bar(['Music', 'Speech', 'Others'], gen_hist[:,genre_list[genre_i]], color='blue', width=0.8)
            plt.ylim([0.5,np.max(gen_hist)])
            plt.title(genre_i)
            plt_count += 1
        # plt.show()
        plt.savefig(PARAMS['opDir']+'/Genre_Wise_thresh'+str(int(PARAMS['high_confidence_thresh']*100))+'.png', bbox_inches='tight')
        print(PARAMS['opDir']+'/Genre_Wise_thresh'+str(int(PARAMS['high_confidence_thresh']*100))+'.png')
        
        plt.figure()
        aud_hist = np.zeros((3,len(genre_list)))
        for genre_i in genre_sp_mu_dist.keys():
            aud_hist[0,genre_list[genre_i]] = genre_sp_mu_dist[genre_i][0]
            aud_hist[1,genre_list[genre_i]] = genre_sp_mu_dist[genre_i][1]
            aud_hist[2,genre_list[genre_i]] = genre_sp_mu_dist[genre_i][2]
        aud_hist = np.divide(aud_hist, np.repeat(np.array(np.sum(aud_hist, axis=0), ndmin=2), np.shape(aud_hist)[0], axis=0))
        # aud_hist = np.divide(aud_hist, np.repeat(np.array(np.sum(aud_hist, axis=1), ndmin=2).T, np.shape(aud_hist)[1], axis=1))
        aud_hist = np.divide(aud_hist, np.repeat(np.array(np.max(aud_hist, axis=1), ndmin=2).T, np.shape(aud_hist)[1], axis=1))
        
        plt.subplot(311)
        plt.bar(genre_sp_mu_dist.keys(), aud_hist[0,:], color='blue', width=0.4)
        plt.ylim([0.5,np.max(aud_hist)])
        plt.title('Music')
    
        plt.subplot(312)
        plt.bar(genre_sp_mu_dist.keys(), aud_hist[1,:], color='maroon', width=0.4)
        plt.ylim([0.5,np.max(aud_hist)])
        plt.title('Speech')
    
        plt.subplot(313)
        plt.bar(genre_sp_mu_dist.keys(), aud_hist[2,:], color='green', width=0.4)
        plt.ylim([0.5,np.max(aud_hist)])
        plt.title('Others')
        # plt.show()
        plt.savefig(PARAMS['opDir']+'/Audio_Wise_thresh'+str(int(PARAMS['high_confidence_thresh']*100))+'.png', bbox_inches='tight')
        print(PARAMS['opDir']+'/Audio_Wise_thresh'+str(int(PARAMS['high_confidence_thresh']*100))+'.png')
    
    
    #%%
    sp_hist = []
    mu_hist = []
    genre_strings = []
    plt.figure()
    plt_count = 0
    hist = np.zeros((len(speech_music),2))
    count = 0
    print('speech_music: ', speech_music)
    for genre_i in speech_music.keys():
        mu_genre = speech_music[genre_i][0]
        sp_genre = speech_music[genre_i][1]
        hist[count,:] = [mu_genre, sp_genre]
        tot_genre = mu_genre+sp_genre
        mu_genre /= tot_genre
        sp_genre /= tot_genre
        mu_hist.append(mu_genre)
        sp_hist.append(sp_genre)
        genre_strings.append(genre_i)
        count += 1
    # hist = np.divide(hist, np.repeat(np.array(np.max(hist, axis=0), ndmin=2), len(speech_music), axis=0))
    hist = np.divide(hist, np.repeat(np.array(np.sum(hist, axis=1), ndmin=2).T, 2, axis=1))
    print(hist)
    
    count = 0
    for genre_i in speech_music.keys():
        plt_count += 1
        plt.subplot(4,4,plt_count)
        plt.bar([1,2], hist[count, :])
        plt.xticks(ticks=[1,2], labels=['Music', 'Speech'])
        plt.title(genre_i)
        # plt.ylim([0,1])
        count += 1
    plt.savefig(PARAMS['opDir']+'/Genre_wise_speech_music'+'.png', bbox_inches='tight')
    
    # mu_hist -= np.mean(mu_hist)
    # mu_hist /= np.std(mu_hist)
    # sp_hist -= np.mean(sp_hist)
    # sp_hist /= np.std(sp_hist)
    print(mu_hist)
    print(sp_hist)
    audio_wise_hist = np.append(np.array(mu_hist, ndmin=2).T, np.array(sp_hist, ndmin=2).T, axis=1)
    savemat(PARAMS['opDir']+'/histograms.mat', {'audio_wise_hist':audio_wise_hist, 'genre_wise_hist':hist})
    

    plt.figure()
    plt.subplot(211)
    plt.bar(list(range(len(genre_strings))), mu_hist)
    plt.xticks(ticks=list(range(len(genre_strings))), labels=genre_strings)
    plt.title('Music')
    plt.ylim([0.7,0.85])
    
    plt.subplot(212)
    plt.bar(list(range(len(genre_strings))), sp_hist)
    plt.xticks(ticks=list(range(len(genre_strings))), labels=genre_strings)
    plt.title('Speech')
    plt.ylim([0.2,0.3])
    
    plt.savefig(PARAMS['opDir']+'/AUdio_wise_speech_music'+'.png', bbox_inches='tight')
    
    print(genre_strings)
    
    