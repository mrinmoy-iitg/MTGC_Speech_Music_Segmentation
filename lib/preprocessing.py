#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:28:27 2019

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import os
from lib.cython.funcs import extract_patches as cextract_patches



def normalize_signal(Xin):
    '''
    Normalize an audio signal by subtracting mean and dividing my the maximum
    amplitude

    Parameters
    ----------
    Xin : array
        Audio signal.

    Returns
    -------
    Xin : array
        Normalized audio signal.

    '''
    Xin = Xin - np.mean(Xin)
    Xin = Xin / np.max(np.abs(Xin))
    return Xin




def get_feature_patches(PARAMS, FV):
    # FV should be of the shape (nFeatures, nFrames)
    if np.shape(FV)[1]<=PARAMS['W']:
        FV1 = FV.copy()
        while np.shape(FV)[1]<=PARAMS['W']:
            FV = np.append(FV, FV1, axis=1)
    
    if len(np.shape(FV))==2: # 'Spec', 'LogSpec', 'MelSpec', 'LogMelSpec'
        FV = FV.T
        FV_scaled = StandardScaler(copy=False).fit_transform(FV)
        FV_scaled = FV_scaled.T
        FV_scaled = np.expand_dims(FV_scaled, axis=2)
    
    elif len(np.shape(FV))==3: # 'MFCC'
        FV_scaled = np.empty([])
        for channel in range(np.shape(FV)[2]):
            FV_channel = np.squeeze(FV[:,:,channel])
            FV_channel = FV_channel.T
            FV_channel_scaled = StandardScaler(copy=False).fit_transform(FV_channel)
            FV_channel_scaled = FV_channel_scaled.T
            if np.size(FV_scaled)<=1:
                FV_scaled = np.expand_dims(FV_channel_scaled, axis=2)
            else:
                FV_scaled = np.append(FV_scaled, np.expand_dims(FV_channel_scaled, axis=2), axis=2)
    patches = cextract_patches(FV_scaled, np.shape(FV_scaled), PARAMS['W'], PARAMS['W_shift'])

    return patches




def load_and_preprocess_signal(fName):
    Xin, fs = librosa.core.load(fName, mono=True, sr=None)
    Xin_norm = normalize_signal(Xin)
    return Xin_norm, fs




def get_featuregram(PARAMS, fName_path, save_feat=True):
    fName = fName_path.split('/')[-1].split('.')[0]

    if not os.path.exists(PARAMS['feature_opDir']+'/'+fName+'.npy'):
        Xin, fs = load_and_preprocess_signal(fName_path)
        if PARAMS['n_fft']<(fs*PARAMS['Tw']/1000):
            n_fft = int(fs*PARAMS['Tw']/1000)
        else:
            n_fft = PARAMS['n_fft']
        frameSize = int(PARAMS['Tw']*fs/1000)
        frameShift = int(PARAMS['Ts']*fs/1000)

        if PARAMS['featName']=='Spec':
            fv = np.abs(librosa.core.stft(y=Xin, n_fft=n_fft, win_length=frameSize, hop_length=frameShift, center=False))
            fv = fv.astype(np.float32)

        if PARAMS['featName']=='LogSpec':
            fv = np.abs(librosa.core.stft(y=Xin, n_fft=n_fft, win_length=frameSize, hop_length=frameShift, center=False))
            fv = librosa.core.power_to_db(fv**2)
            fv = fv.astype(np.float32)

        elif PARAMS['featName']=='MelSpec':
            fv = librosa.feature.melspectrogram(y=Xin, sr=fs, n_fft=n_fft, win_length=frameSize, hop_length=frameShift, center=False, n_mels=PARAMS['n_mels'])
            fv = fv.astype(np.float32)

        elif PARAMS['featName']=='LogMelSpec':
            fv = librosa.feature.melspectrogram(y=Xin, sr=fs, n_fft=n_fft, win_length=frameSize, hop_length=frameShift, center=False, n_mels=PARAMS['n_mels'])
            fv = librosa.core.power_to_db(fv**2)
            fv = fv.astype(np.float32)

        elif PARAMS['featName']=='MFCC':
            fv = librosa.feature.mfcc(y=Xin, sr=fs, n_mfcc=PARAMS['n_mfcc'], n_fft=n_fft, win_length=frameSize, hop_length=frameShift, center=False, n_mels=PARAMS['n_mels'])
            fv_delta = librosa.feature.delta(fv, width=3, order=1)
            fv_delta_delta = librosa.feature.delta(fv, width=3, order=2)
            fv = np.expand_dims(fv, axis=2)
            fv = np.append(fv, np.expand_dims(fv_delta, axis=2), axis=2)
            fv = np.append(fv, np.expand_dims(fv_delta_delta, axis=2), axis=2)
            fv = fv.astype(np.float32)

        if save_feat:
            np.save(PARAMS['feature_opDir']+'/'+fName+'.npy', fv)
    else:
        fv = np.load(PARAMS['feature_opDir']+'/'+fName+'.npy', allow_pickle=True)
    
    return fv

