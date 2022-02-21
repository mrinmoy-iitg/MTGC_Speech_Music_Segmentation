#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 18:11:28 2021

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import lib.spectral_peak_tracking as SPT
import datetime
import os
import librosa.display
import lib.misc as misc
import pickle
from scipy import stats



def computeSPT_Stats(FV, derivative=False):
    Statistics = np.empty([])
    for seg_i in range(np.shape(FV)[0]):
        sequences = np.squeeze(FV[seg_i, :, :])
        if derivative:
            sequences = np.diff(sequences, axis=0)
            sequences -= np.min(sequences)
            sequences += 1e-10
        stats_seq = np.empty([])
        mean = np.array(np.mean(sequences, axis=0), ndmin=2)
        stats_seq = mean
        # print(f'mean: {np.shape(mean)}')
        gmean = np.array(stats.gmean(sequences, axis=0), ndmin=2)
        stats_seq = np.append(stats_seq, gmean, axis=1)
        # print(f'gmean: {np.shape(gmean)}')
        gstd = np.array(stats.gstd(sequences, axis=0), ndmin=2)
        stats_seq = np.append(stats_seq, gstd, axis=1)
        # print(f'gstd: {np.shape(gstd)}')
        hmean = np.array(stats.hmean(sequences, axis=0), ndmin=2)
        stats_seq = np.append(stats_seq, hmean, axis=1)
        # print(f'hmean: {np.shape(hmean)}')
        entr = np.array(stats.entropy(sequences, axis=0), ndmin=2)
        stats_seq = np.append(stats_seq, entr, axis=1)
        # print(f'entr: {np.shape(entr)}')
        med = np.array(np.median(sequences, axis=0), ndmin=2)
        stats_seq = np.append(stats_seq, med, axis=1)
        # print(f'med: {np.shape(med)}')
        mode, count = stats.mode(sequences, axis=0)
        stats_seq = np.append(stats_seq, mode, axis=1)
        # print(f'mode: {np.shape(mode)}')
        stdev = np.array(np.std(sequences, axis=0), ndmin=2)
        stats_seq = np.append(stats_seq, stdev, axis=1)
        # print(f'stdev: {np.shape(stdev)}')
        maxx = np.array(np.max(sequences, axis=0), ndmin=2)
        stats_seq = np.append(stats_seq, maxx, axis=1)
        # print(f'maxx: {np.shape(maxx)}')
        minn = np.array(np.max(sequences, axis=0), ndmin=2)
        stats_seq = np.append(stats_seq, minn, axis=1)
        # print(f'minn: {np.shape(minn)}')
        kurt = np.array(stats.kurtosis(sequences, axis=0), ndmin=2)
        stats_seq = np.append(stats_seq, kurt, axis=1)
        # print(f'kurt: {np.shape(kurt)}')
        skew = np.array(stats.skew(sequences, axis=0), ndmin=2)
        stats_seq = np.append(stats_seq, skew, axis=1)
        # print(f'skew: {np.shape(skew)}')
        # print(f'stats_seq: {np.shape(stats_seq)}')
        # print('msd: ', np.shape(msd), np.shape(sequences))
        if np.size(Statistics)<=1:
            Statistics = stats_seq
        else:
            Statistics = np.append(Statistics, stats_seq, axis=0)
            
    return Statistics





def computeSPT_MeanStd(FV):
    Peak_MeanStd = np.empty([])
    for seg_i in range(np.shape(FV)[0]):
        mn = np.mean(np.squeeze(FV[seg_i, :, :]), axis=0)
        sd = np.std(np.squeeze(FV[seg_i, :, :]), axis=0)
        msd = np.array(np.append(mn, sd), ndmin=2)
        # print('msd: ', np.shape(msd), np.shape(np.squeeze(FV[seg_i, :, :])))
        if np.size(Peak_MeanStd)<=1:
            Peak_MeanStd = msd
        else:
            Peak_MeanStd = np.append(Peak_MeanStd, msd, axis=0)
            
    return Peak_MeanStd




''' This function computes posterior probability features ''' 
def get_gmm_posterior_features(PARAMS, FV, feat_type):
    data_lkhd = np.empty([])
    
    ''' Generating the likelihood features from the learned GMMs '''
    print('Generating likelihood features')
    
    for seg_i in range(np.shape(FV)[0]):
        lkhd = np.empty([])
        for seq_i in range(np.shape(FV)[2]):
            seq = np.array(np.squeeze(FV[seg_i, :, seq_i]), ndmin=2).T
            # print('seq: ', np.shape(seq), np.shape(FV))
            lkhd_seq = np.empty([])
            for clNum in PARAMS['classes'].keys():
                gmm = None
                ''' Load already learned GMM '''
                optFileGMM = PARAMS['gmmPath_'+feat_type] + '/fold' + str(PARAMS['SM_classifier_fold']) + '/Cl' + str(clNum) + '_seq' + str(seq_i) + '_train_data_gmm.pkl'
                with open(optFileGMM, 'rb') as file:  
                    gmm = pickle.load(file)
    
                proba = gmm.predict_proba(seq)
                mean_idx = np.ndarray.argsort(np.squeeze(gmm.means_))
                proba = proba[:, mean_idx]
                proba_fv = np.mean(proba, axis=0)
                
                if np.size(lkhd_seq)<=1:
                    lkhd_seq = proba_fv
                else:
                    lkhd_seq = np.append(lkhd_seq, proba_fv)
                # print('lkhd_seq: ', np.shape(lkhd_seq))
            if np.size(lkhd)<=1:
                lkhd = np.array(lkhd_seq, ndmin=2)
            else:
                lkhd = np.append(lkhd, np.array(lkhd_seq, ndmin=2), axis=1)
            # print('lkhd: ', np.shape(lkhd))
        
        if np.size(data_lkhd)<=1:
            data_lkhd = lkhd
        else:
            data_lkhd = np.append(data_lkhd, lkhd, axis=0)
        # print('data_lkhd: ', np.shape(data_lkhd))

    print('Likelihood features computed')
    return data_lkhd



def get_SPT_matrices(opDir_SPT, fl):
    if not os.path.exists(opDir_SPT+'/'+fl.split('/')[-1].split('.')[0]+'.pkl'):
        Xin, fs = librosa.core.load(fl, mono=True, sr=16000) # Keeping sampling rate same as MUSAN
        print('Xin: ', np.shape(Xin), fs)
        frameSize = int((PARAMS['Tw']*fs)/1000) # Frame size in number of samples
        frameShift = int((PARAMS['Ts']*fs)/1000) # Frame shift in number of samples
        LSPT, ASPT, stft, peak_repeat_count = SPT.computeSPT(y=Xin, sr=fs, frame_length=frameSize, hop_length=frameShift, req_pks=PARAMS['nTopPeaks'])
        print('Peak tracking: ASPT=', np.shape(ASPT), ' LSPT=', np.shape(LSPT))
        numSamplesInInterval = int(PARAMS['intervalSize']*fs/1000)
        numSamplesShiftPerInterval = int(PARAMS['intervalShift']*fs/1000)
        numFv = int(np.floor((len(Xin)-numSamplesInInterval)/numSamplesShiftPerInterval))+1
        nFrames = np.shape(ASPT)[0]

        peak_repeat_statistics_file = PARAMS['opDir'] + '/Peak_repeat_statistics.csv'
        fid = open(peak_repeat_statistics_file, 'a+', encoding='utf8')
        fid.write(fl.split('/')[-1] + '\t' + str(peak_repeat_count) + '\t' + str(nFrames) + '\n')
        fid.close()
        
        nmFrmPerInterval = int(np.floor((numSamplesInInterval-frameSize)/frameShift))+1
        nmFrmPerIntervalShift = int(np.floor((numSamplesShiftPerInterval-frameSize)/frameShift))+1
        print('Signal: ', len(Xin), numSamplesInInterval, numFv, nFrames, nmFrmPerInterval, np.shape(ASPT))
        
        frmStart = 0
        frmEnd = 0
        fv_val = np.empty([])
        fv_loc = np.empty([])
        for l in range(numFv):
            frmStart = l*nmFrmPerIntervalShift
            frmEnd = l*nmFrmPerIntervalShift + nmFrmPerInterval
            if frmEnd>nFrames:
                frmEnd = nFrames
                frmStart = frmEnd-nmFrmPerInterval
            val = np.array(ASPT[frmStart:frmEnd, :], ndmin=2)
            val = np.expand_dims(val, axis=0)
            loc = np.array(LSPT[frmStart:frmEnd, :], ndmin=2)
            loc = np.expand_dims(loc, axis=0)
            if np.size(fv_val)<=1:
                fv_val = val
                fv_loc = loc
            else:
                fv_val = np.append(fv_val, val, 0)
                fv_loc = np.append(fv_loc, loc, 0)

        SPT_data = {'val': fv_val, 'loc': fv_loc}
        misc.save_obj(SPT_data, opDir_SPT, fl.split('/')[-1].split('.')[0])
        print('APT: ', np.shape(fv_val), ' LPT: ', np.shape(fv_loc), ' Sampling rate: ', fs)

    else:
        fv_val = misc.load_obj(opDir_SPT, fl.split('/')[-1].split('.')[0])['val']
        fv_loc = misc.load_obj(opDir_SPT, fl.split('/')[-1].split('.')[0])['loc']
        print('APT: ', np.shape(fv_val), ' LPT: ', np.shape(fv_loc))
    
    return fv_val, fv_loc



def __init__():
    PARAMS = {
        'Tw': 10, # Frame size in miliseconds
        'Ts': 5, # Frame shift in miliseconds
        'nTopPeaks': 10, # Number of Prominent peaks to be considered for SPS computation
        'silThresh': 0.0,  # Silence Threshold
        'intervalSize': 1000, # In miliseconds
        'intervalShift': 1000, # In miliseconds
        'dataset_name': 'Moviescope',
        'audio_path': '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/Moviescope/', 
        'gmmPath_ASPT': '/home/mrinmoy/Documents/PhD_Work/Features/MTGC_SMO/musan/1000ms_10ms_5ms_10PT/CBoW-ASPT_mix5/__GMMs/',
        'gmmPath_LSPT': '/home/mrinmoy/Documents/PhD_Work/Features/MTGC_SMO/musan/1000ms_10ms_5ms_10PT/CBoW-LSPT_mix5/__GMMs/',
        'output_path': '/home/mrinmoy/Documents/PhD_Work/Features/MTGC_SMO/',        
        'today': datetime.datetime.now().strftime("%Y-%m-%d"),
        'classes':{0:'music', 1:'speech'},
        'SM_classifier_fold': 0,
        }

    return PARAMS




if __name__ == '__main__':
    PARAMS = __init__()
    print(PARAMS['today'])

    PARAMS['opDir'] = PARAMS['output_path'] + '/' + PARAMS['dataset_name'] + '/1s_1s_' + str(PARAMS['Tw'])+'ms_'+str(PARAMS['Ts']).replace('.','-')+'ms_'+ str(PARAMS['nTopPeaks']) +'PT_' + PARAMS['today']
    if not os.path.exists(PARAMS['opDir']):
        os.makedirs(PARAMS['opDir'])

    misc.print_configuration(PARAMS)
    
    opDir_SPT = PARAMS['opDir'] + '/SPT/'
    if not os.path.exists(opDir_SPT):
        os.makedirs(opDir_SPT)
        
    opDirMSD_ASPT_LSPT = PARAMS['opDir'] + '/MSD-ASPT-LSPT/'
    if not os.path.exists(opDirMSD_ASPT_LSPT):
        os.makedirs(opDirMSD_ASPT_LSPT)

    opDirCBoW_ASPT_LSPT = PARAMS['opDir'] + '/CBoW-ASPT-LSPT/'
    if not os.path.exists(opDirCBoW_ASPT_LSPT):
        os.makedirs(opDirCBoW_ASPT_LSPT)

    opDirStats_ASPT_LSPT = PARAMS['opDir'] + '/Stats-ASPT-LSPT/'
    if not os.path.exists(opDirStats_ASPT_LSPT):
        os.makedirs(opDirStats_ASPT_LSPT)
        
    # opDirDtvStats_ASPT_LSPT = PARAMS['opDir'] + '/Derivative_Stats-ASPT-LSPT/'
    # if not os.path.exists(opDirDtvStats_ASPT_LSPT):
    #     os.makedirs(opDirDtvStats_ASPT_LSPT)

    path = PARAMS['audio_path'] + '/wav/' 	#path where the audio is
    print('\n\n',path)
    files = librosa.util.find_files(path, ext=['wav'])

    fileCount = 0 #to see in the terminal how many files have been loaded
    for fl in files:
        fileCount += 1
        print('\n', fileCount, ' Audio file: ', fl.split('/')[-1])
        
        msd_aspt_lspt_fName = opDirMSD_ASPT_LSPT + fl.split('/')[-1].split('.')[0] + '.npy'
        cbow_aspt_lspt_fName = opDirCBoW_ASPT_LSPT + fl.split('/')[-1].split('.')[0] + '.npy'
        stats_aspt_lspt_fName = opDirStats_ASPT_LSPT + fl.split('/')[-1].split('.')[0] + '.npy'
        # derivative_stats_aspt_lspt_fName = opDirDtvStats_ASPT_LSPT + fl.split('/')[-1].split('.')[0] + '.npy'
        
        if not os.path.exists(msd_aspt_lspt_fName):
            fv_val, fv_loc = get_SPT_matrices(opDir_SPT, fl)            
            MSD_LSPT = computeSPT_MeanStd(fv_loc)
            MSD_ASPT = computeSPT_MeanStd(fv_val)
            print('MSD features: ', np.shape(MSD_ASPT), np.shape(MSD_LSPT))
            MSD_ASPT_LSPT = np.append(MSD_ASPT, MSD_LSPT, axis=1)
            print('MSD_ASPT_LSPT: ', np.shape(MSD_ASPT_LSPT))
            np.save(msd_aspt_lspt_fName, MSD_ASPT_LSPT)
        else:
            MSD_ASPT_LSPT = np.load(msd_aspt_lspt_fName, allow_pickle=True)
            print('MSD_ASPT_LSPT: ', np.shape(MSD_ASPT_LSPT))
            
        if not os.path.exists(cbow_aspt_lspt_fName):
            fv_val, fv_loc = get_SPT_matrices(opDir_SPT, fl)            
            CBoW_ASPT = get_gmm_posterior_features(PARAMS, fv_val, 'ASPT')
            CBoW_LSPT = get_gmm_posterior_features(PARAMS, fv_loc, 'LSPT')
            print('CBoW features: ', np.shape(CBoW_ASPT), np.shape(CBoW_LSPT))
            CBoW_ASPT_LSPT = np.append(CBoW_ASPT, CBoW_LSPT, axis=1)
            print('CBoW-ASPT-LSPT: ', np.shape(CBoW_ASPT_LSPT))
            np.save(cbow_aspt_lspt_fName, CBoW_ASPT_LSPT)
        else:
            CBoW_ASPT_LSPT = np.load(cbow_aspt_lspt_fName, allow_pickle=True)
            print('CBoW_ASPT_LSPT: ', np.shape(CBoW_ASPT_LSPT))

        if not os.path.exists(stats_aspt_lspt_fName):
            fv_val, fv_loc = get_SPT_matrices(opDir_SPT, fl)            
            Stats_LSPT = computeSPT_Stats(fv_loc)
            Stats_ASPT = computeSPT_Stats(fv_val)
            print('Stats features: ', np.shape(Stats_ASPT), np.shape(Stats_LSPT))
            Stats_ASPT_LSPT = np.append(Stats_ASPT, Stats_LSPT, axis=1)
            print('Stats_ASPT_LSPT: ', np.shape(Stats_ASPT_LSPT))
            np.save(stats_aspt_lspt_fName, Stats_ASPT_LSPT)
        else:
            Stats_ASPT_LSPT = np.load(stats_aspt_lspt_fName, allow_pickle=True)
            print('Stats_ASPT_LSPT: ', np.shape(Stats_ASPT_LSPT))

        # if not os.path.exists(derivative_stats_aspt_lspt_fName):
        #     fv_val, fv_loc = get_SPT_matrices(opDir_SPT, fl)            
        #     Dtv_Stats_LSPT = computeSPT_Stats(fv_loc, derivative=True)
        #     Dtv_Stats_ASPT = computeSPT_Stats(fv_val, derivative=True)
        #     print('Derivative Stats features: ', np.shape(Dtv_Stats_ASPT), np.shape(Dtv_Stats_LSPT))
        #     Dtv_Stats_ASPT_LSPT = np.append(Dtv_Stats_ASPT, Dtv_Stats_LSPT, axis=1)
        #     print('Derivative Stats_ASPT_LSPT: ', np.shape(Dtv_Stats_ASPT_LSPT))
        #     np.save(derivative_stats_aspt_lspt_fName, Dtv_Stats_ASPT_LSPT)
        # else:
        #     Dtv_Stats_ASPT_LSPT = np.load(derivative_stats_aspt_lspt_fName, allow_pickle=True)
        #     print('Derivative Stats_ASPT_LSPT: ', np.shape(Dtv_Stats_ASPT_LSPT))
