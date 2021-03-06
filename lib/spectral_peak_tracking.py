#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 23:03:44 2019

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati

"""

import numpy as np
import librosa
import scipy.signal
from joblib import Parallel, delayed
import multiprocessing
import scipy.stats




def peakFinding(seq, req_pks, n_fft):
    loc = np.array(scipy.signal.find_peaks(x=seq))
    loc = loc[0]
    hgt = np.array(seq[loc])
    peak_repeat = False
    if len(hgt) < req_pks:
        hgt = np.append(hgt, np.ones([req_pks-len(hgt),1],int)*hgt[len(hgt)-1])
        loc = np.append(loc, np.ones([req_pks-len(loc),1],int)*loc[len(loc)-1])
        peak_repeat = True
    # Getting the top peaks in sorted order
    hgtIdx = np.argsort(hgt)[::-1][:req_pks]
    pkIdx = np.sort(hgtIdx)
    # Normalized over the whole energy of the frame
    rms_energy = np.sqrt(np.mean(np.power(seq,2)))
    if rms_energy==0:
        rms_energy = 1
    amplitudes = np.array(hgt[pkIdx]/rms_energy, ndmin=2)
    
    # Normalized over the sampling rate
    norm_freq_bins = np.array(loc[pkIdx]/n_fft, ndmin=2)

    return amplitudes, norm_freq_bins, peak_repeat



def computeSPT_Parallel(y, sr, frame_length, hop_length, req_pks):
    window = np.hamming(frame_length)
    stft = librosa.stft(y=y, n_fft=frame_length, hop_length=hop_length, window=window)
    stft = 2 * np.abs(stft) / np.sum(window)
    nFrames = stft.shape[1]
    PkLoc = np.array
    PkVal = np.array

    num_cores = multiprocessing.cpu_count()-1
    peaks = Parallel(n_jobs=num_cores)(delayed(peakFinding)(seq=stft[:,k], req_pks=req_pks, n_fft=frame_length) for k in range(nFrames))
    PkVal, PkLoc, peak_repeat = zip(*peaks)
    
    PkVal = np.array(np.squeeze(PkVal), ndmin=2)
    PkLoc = np.array(np.squeeze(PkLoc), ndmin=2)
    peak_repeat_count = np.sum(peak_repeat)

    if req_pks==1:
        PkVal = np.transpose(PkVal)
        PkLoc = np.transpose(PkLoc)
        
    return PkLoc, PkVal, stft, peak_repeat_count





def computeSPT(y, sr, frame_length, hop_length, req_pks):
    '''
    Function to compute Spectral Peak Tracking (SPT) Matrix
    Usage:
          [PkLoc, PkVal] = computeSPT(y, sr, n_fft, hop_length, req_pks)
    Input:
          y: Audio signal
          sr : Sampling rate
          frame_length : Frame size (in no. of samples)
          hop_length : Frame shift (in no. of samples)
          req_pks : Number of prominent peaks required
    Output:
          PkLoc : Peak Frequencies
          PkVal : Peak Amplitudes
    '''
    window = np.hamming(frame_length)
    stft = librosa.stft(y=y, n_fft=frame_length, hop_length=hop_length, window=window)
    stft = 2 * np.abs(stft) / np.sum(window)
    nFrames = stft.shape[1]
    PkLoc = np.array
    PkVal = np.array
    peak_repeat_count = 0
    print('stft: ', np.shape(stft), nFrames)

    for k in range(nFrames):
        frmFt = stft[:,k]
        peaks, props = scipy.signal.find_peaks(x=frmFt)
        loc = np.array(peaks)
        hgt = np.array(frmFt[loc])
        
        # print('hgt: ', len(hgt.tolist()))
        if len(hgt.tolist())==0:
            continue
        '''
        Repeating the highest frequency peak to keep same cardinality in every frame
        '''
        if(len(hgt) < req_pks):
            hgt = np.append(hgt, np.ones([req_pks-len(hgt),1],int)*hgt[len(hgt)-1])
            loc = np.append(loc, np.ones([req_pks-len(loc),1],int)*loc[len(loc)-1])
            peak_repeat_count += 1
        
        hgtIdx = np.argsort(hgt)[::-1][:req_pks]# Getting the top peaks in sorted order
        pkIdx = np.sort(hgtIdx)
        
        amplitudes = np.array(hgt[pkIdx]/np.sqrt(np.mean(np.power(frmFt,2))), ndmin=2)# Normalized over the whole energy of the frame
        norm_freq_bins = np.array(loc[pkIdx]/frame_length, ndmin=2)# Normalized over the sampling rate
    
        if(np.size(PkVal)==1):
            PkVal = np.array(amplitudes, ndmin=2)
            PkLoc = np.array(norm_freq_bins, ndmin=2)
        else:
            PkVal = np.append(PkVal, np.array(amplitudes, ndmin=2), 0)
            PkLoc = np.append(PkLoc, np.array(norm_freq_bins, ndmin=2), 0)
                
    return PkLoc, PkVal, stft, peak_repeat_count



def parallel_SPT_MeanStd(SPPS):
    time_mean = np.mean(SPPS, axis=0)
    MeanStd = np.array(time_mean, ndmin=2)

    time_std = np.std(SPPS, axis=0)
    MeanStd = np.append(MeanStd, np.array(time_std, ndmin=2), 1)

    return MeanStd


def computeSPT_MeanStd(numFv, nmFrmPerInterval, nmFrmPerIntervalShift, nFrames, PkLoc):
    frmStart = -1
    frmEnd = 0
    Peak_MeanStd = np.array
    print('NumFV: ', numFv, ' NumFrmPerSec: ', nmFrmPerInterval, ' nFrames: ', nFrames, ' PkLocShape: ', np.shape(PkLoc))

    num_cores = multiprocessing.cpu_count()
    frmStart = list(range(0,nFrames,nmFrmPerIntervalShift))
    frmEnd = (np.array(frmStart)+nmFrmPerInterval).tolist()
    if frmEnd[-1]>nFrames:
        frmEnd[-1] = nFrames
        frmStart[-1] = frmEnd[-1] - nmFrmPerInterval
    print('Length: frmStart=', len(frmStart), ' frmEnd=', len(frmEnd), numFv)
    MeanStd = Parallel(n_jobs=num_cores)(delayed(parallel_SPT_MeanStd)(SPPS=PkLoc[frmStart[l]:frmEnd[l],:]) for l in range(numFv))
    MeanStd = np.array(MeanStd)
    Peak_MeanStd = np.array(np.squeeze(MeanStd), ndmin=2)

    return Peak_MeanStd
