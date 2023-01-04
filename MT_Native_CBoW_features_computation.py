#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 20:48:54 2022

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import lib.spectral_peak_tracking as SPT
import datetime
import os
import librosa.display
import lib.misc as misc
import pickle
import sys
from sklearn import mixture
from scipy.signal import medfilt
from scipy import interpolate
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
                optFileGMM = PARAMS['gmmPath_'+feat_type] + '/Cl' + str(clNum) + '_seq' + str(seq_i) + '_train_data_gmm.pkl'
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




'''
This function learns GMMs by taking all data of any i^th sequence at a time. 
Finally, k GMMs are learned, where k is the number of sequences in the the SPS matrix
''' 
def learn_seq_gmm(PARAMS, seq_data, optFileGMM):
    flattened_seq_data = np.transpose(np.array(seq_data.flatten(), ndmin=2))
    print('flattened_seq_data: ', np.shape(flattened_seq_data))
    gmm = mixture.GaussianMixture(n_components=PARAMS['numMix'], covariance_type='full', verbose=1)
    gmm.fit(flattened_seq_data.astype(np.float32))
    print('Saving GMM model as ', optFileGMM)
    with open(optFileGMM, 'wb') as file:
        pickle.dump(gmm, file)




'''
This function reads in the files listed and returns their contents as a single
2D array
'''
def load_files_cbow_features(PARAMS, files, seqNum, feat_type, audio_type):
    data = np.empty([],dtype=float)
    file_mark = np.empty([],dtype=float)
    data_file_mark = {}
    
    fileCount = 0
    data_marking_start = 0
    data_marking = 0
    for fl in files:
        fileCount += 1
        # fName = data_path + '/' + fl
        # SPS_matrix = np.load(fName, allow_pickle=True)
        if not os.path.exists(PARAMS['opDir_SPT']+'/'+fl.split('.')[0]+'.pkl'):
            continue
        if feat_type=='aspt':
            SPS_matrix = misc.load_obj(PARAMS['opDir_SPT'], fl.split('.')[0])['val']
        elif feat_type=='lspt':
            SPS_matrix = misc.load_obj(PARAMS['opDir_SPT'], fl.split('.')[0])['loc']
        
        sm_pred = np.load(PARAMS['sm_pred_path']+'/'+fl.split('.')[0]+'.npy')
        ''' Smoothing SM Predictions '''
        sm_pred_smooth = np.zeros((np.shape(SPS_matrix)[0],2))
        pred_smooth_med = medfilt(sm_pred[:,1], kernel_size=3)
        f = interpolate.interp1d(np.linspace(0,1,len(pred_smooth_med)), pred_smooth_med)
        sm_pred_smooth[:,1] = f(np.linspace(0,1,np.shape(SPS_matrix)[0]))
        sm_pred_smooth[:,0] = 1 - sm_pred_smooth[:,1]
        sm_pred = sm_pred_smooth.copy()

        if len(PARAMS['classes'])==2:
            segment_labels = (sm_pred[:,1]>0.5).astype(int)
        elif len(PARAMS['classes'])==3:
            segment_labels = np.ones(np.shape(sm_pred)[0])*2
            segment_labels[sm_pred[:,1]>=0.75] = 1
            segment_labels[sm_pred[:,1]<=0.25] = 0
        if audio_type=='music':
            idx = (segment_labels==0).astype(int)
        elif audio_type=='speech':
            idx = (segment_labels==1).astype(int)
        elif audio_type=='others':
            idx = (segment_labels==2).astype(int)
        if np.asarray(idx).size==0:
            continue
        # print('idx: ', idx)
        row = SPS_matrix[idx, :, seqNum]
        print(f'({fileCount:4d}/{len(files):4d}) {fl:8} {audio_type:6} SPS matrix={np.shape(SPS_matrix)}, sm_pred: {np.shape(sm_pred)}, row shape: {np.shape(row)}', end='\r', flush=True)
        
        markings = np.ones((1,np.shape(row)[0]))
        markings *= np.shape(row)[1]
        markings = markings.astype(int)
        row = row.ravel()
        data_marking += np.shape(SPS_matrix)[0]
        
        '''
        Storing the row in data array.
        '''
        if np.size(data)<=1:
            data = np.array(row, ndmin=2)
            file_mark = np.array(markings, ndmin=2)
        else:
            data = np.append(data, np.array(row, ndmin=2), 1)
            file_mark = np.append(file_mark, np.array(markings, ndmin=2), 1)
        
        data_file_mark.setdefault(fl,[]).append([data_marking_start, data_marking])
        data_marking_start = data_marking
        
    file_mark = np.cumsum(file_mark).astype(int)
    
    print('\nData loaded: ', np.shape(data), np.shape(file_mark))
    return data, file_mark, data_file_mark




def compute_SPT_matrices(PARAMS, files):
    fl_count = 0
    for fl in files:
        fl_count += 1
        if os.path.exists(PARAMS['opDir_SPT']+'/'+fl.split('/')[-1].split('.')[0]+'.pkl'):
            print(f'\n({fl_count}/{len(files)}) {fl} SPT matrix already computed')
            continue
        if not os.path.exists(PARAMS['audio_path']+'/'+fl):
            print(f'\n({fl_count}/{len(files)}) {fl} audio file does not exists')
            continue
        Xin, fs = librosa.core.load(PARAMS['audio_path']+'/'+fl, mono=True, sr=None) # Keeping sampling rate same as MUSAN
        print(f'\n({fl_count}/{len(files)}) {fl} Xin: {np.shape(Xin)}, {fs}')
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
        print(f'Signal: {len(Xin)}, {numSamplesInInterval}, {numFv}, {nFrames}, {nmFrmPerInterval}, {np.shape(ASPT)}')
        
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
        misc.save_obj(SPT_data, PARAMS['opDir_SPT'], fl.split('/')[-1].split('.')[0])
        print(f'APT: {np.shape(fv_val)} LPT: {np.shape(fv_loc)} Sampling rate: {fs}')
    



def compute_features(PARAMS, files, file_type):
    fileCount = 0 #to see in the terminal how many files have been loaded
    for fl in files:
        fileCount += 1
        fName = fl.split('/')[-1]
        print(f'\n{file_type} {fileCount} Audio file: {fName}')
        if not os.path.exists(PARAMS['opDir_SPT']+'/'+fName.split('.')[0]+'.pkl'):
            continue
        
        msd_aspt_lspt_fName = PARAMS['opDirMSD_ASPT_LSPT'] + fName.split('.')[0] + '.npy'
        cbow_aspt_lspt_fName = PARAMS['opDirCBoW_ASPT_LSPT'] + fName.split('.')[0] + '.npy'
        stats_aspt_lspt_fName = PARAMS['opDirStats_ASPT_LSPT'] + fName.split('.')[0] + '.npy'
        if not os.path.exists(stats_aspt_lspt_fName):
            fv_val = misc.load_obj(PARAMS['opDir_SPT'], fName.split('.')[0])['val']
            fv_loc = misc.load_obj(PARAMS['opDir_SPT'], fName.split('.')[0])['loc']
            print('APT: ', np.shape(fv_val), ' LPT: ', np.shape(fv_loc))
        
            MSD_ASPT = computeSPT_MeanStd(fv_val)
            MSD_LSPT = computeSPT_MeanStd(fv_loc)
            print(f'MSD features: {np.shape(MSD_ASPT)}, {np.shape(MSD_LSPT)}')
            MSD_ASPT_LSPT = np.append(MSD_ASPT, MSD_LSPT, axis=1)
            print(f'MSD_ASPT_LSPT: {np.shape(MSD_ASPT_LSPT)}')
            np.save(msd_aspt_lspt_fName, MSD_ASPT_LSPT)
    
            CBoW_ASPT = get_gmm_posterior_features(PARAMS, fv_val, 'aspt')
            CBoW_LSPT = get_gmm_posterior_features(PARAMS, fv_loc, 'lspt')
            print(f'CBoW features: {np.shape(CBoW_ASPT)}, {np.shape(CBoW_LSPT)}')
            CBoW_ASPT_LSPT = np.append(CBoW_ASPT, CBoW_LSPT, axis=1)
            print(f'CBoW-ASPT-LSPT: {np.shape(CBoW_ASPT_LSPT)}')
            np.save(cbow_aspt_lspt_fName, CBoW_ASPT_LSPT)

            Stats_ASPT = computeSPT_Stats(fv_val)
            Stats_LSPT = computeSPT_Stats(fv_loc)
            print(f'Stats features: {np.shape(Stats_ASPT)}, {np.shape(Stats_LSPT)}')
            Stats_ASPT_LSPT = np.append(Stats_ASPT, Stats_LSPT, axis=1)
            print(f'Stats_ASPT_LSPT: {np.shape(Stats_ASPT_LSPT)}')
            np.save(stats_aspt_lspt_fName, Stats_ASPT_LSPT)
        else:
            MSD_ASPT_LSPT = np.load(msd_aspt_lspt_fName, allow_pickle=True)
            print(f'MSD_ASPT_LSPT: {np.shape(MSD_ASPT_LSPT)}')
            CBoW_ASPT_LSPT = np.load(cbow_aspt_lspt_fName, allow_pickle=True)
            print(f'CBoW_ASPT_LSPT: {np.shape(CBoW_ASPT_LSPT)}')
            Stats_ASPT_LSPT = np.load(stats_aspt_lspt_fName, allow_pickle=True)
            print(f'Stats_ASPT_LSPT: {np.shape(Stats_ASPT_LSPT)}')
    





def __init__():
    PARAMS = {
        'Tw': 10, # Frame size in miliseconds
        'Ts': 5, # Frame shift in miliseconds
        'nTopPeaks': 5, # Number of Prominent peaks to be considered for SPS computation
        'intervalSize': 1000, # In miliseconds
        'intervalShift': 1000, # In miliseconds
        'numMix': 5,
        
        # Laptop
        # 'dataset_name': 'EmoGDB',
        # 'audio_path': '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/EmoGDB/', 
        # 'gmmPath_ASPT': '/home/mrinmoy/Documents/PhD_Work/Features/MTGC_SMO/musan/1000ms_10ms_5ms_10PT/CBoW-ASPT_mix5/__GMMs/',
        # 'gmmPath_LSPT': '/home/mrinmoy/Documents/PhD_Work/Features/MTGC_SMO/musan/1000ms_10ms_5ms_10PT/CBoW-LSPT_mix5/__GMMs/',
        # 'output_path': '/home/mrinmoy/Documents/PhD_Work/Features/MTGC_SMO/',

        # LabPC
        # 'dataset_name': 'movie-trailers-dataset-master',
        # 'audio_path': '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/movie-trailers-dataset-master/',
        # 'gmmPath_ASPT': '/home/mrinmoy/Documents/PhD_Work/Features/MTGC_SMO/musan/1000ms_10ms_5ms_10PT/CBoW-ASPT_mix5/__GMMs/',
        # 'gmmPath_LSPT': '/home/mrinmoy/Documents/PhD_Work/Features/MTGC_SMO/musan/1000ms_10ms_5ms_10PT/CBoW-LSPT_mix5/__GMMs/',
        # 'output_path': '/home/mrinmoy/Documents/PhD_Work/Features/MTGC_SMO/',

        # LabPC
        # 'dataset_name': 'Moviescope',
        # 'audio_path': '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/Moviescope/wav/', 
        # 'output_path': '/home/mrinmoy/Documents/PhD_Work/Features/MTGC_SMO/',
        # 'sm_pred_path': '/home/mrinmoy/Documents/PhD_Work/Features/MTGC_SMO/Moviescope/1s_1s_10ms_5ms_10PT_2022-01-01/SpMu_Predictions_CBoW-ASPT-LSPT/',
        
        # DGX server
        'dataset_name': 'Moviescope',
        'audio_path': '/workspace/pguha_pg/Mrinmoy/data/Moviescope_dummy/wav/', 
        'output_path': '/workspace/pguha_pg/Mrinmoy/features/MTGC_SMO/',
        'sm_pred_path': '/workspace/pguha_pg/Mrinmoy/features/MTGC_SMO/Moviescope/1s_1s_10ms_5ms_10PT_2022-01-21/SpMu_Predictions_CBoW-ASPT-LSPT/',
        
        'today': datetime.datetime.now().strftime("%Y-%m-%d"),
        'classes':{0:'music', 1:'speech'}, # , 2:'others'
        }

    PARAMS['CV_path_train'] = './cross_validation_info/' + PARAMS['dataset_name'] + '/'    
    if not os.path.exists(PARAMS['CV_path_train']+'/cv_file_list_original.pkl'):
        print('Fold division of files for cross-validation not done!')
        sys.exit(0)
    PARAMS['cv_file_list'] = misc.load_obj(PARAMS['CV_path_train'], 'cv_file_list_original')

    PARAMS['train_files'] = {'wav':PARAMS['cv_file_list']['original_splits']['train']}
    PARAMS['val_files'] = {'wav':PARAMS['cv_file_list']['original_splits']['val']}
    PARAMS['test_files'] = {'wav':PARAMS['cv_file_list']['original_splits']['test']}
    
    PARAMS['opDir'] = PARAMS['output_path'] + '/' + PARAMS['dataset_name'] + '/Native_CBoW_Features_' + str(len(PARAMS['classes'])) + 'classes_' + str(PARAMS['nTopPeaks']) + 'PT_' + str(PARAMS['numMix']) + 'mix_' + PARAMS['today']
    if not os.path.exists(PARAMS['opDir']):
        os.makedirs(PARAMS['opDir'])

    return PARAMS




if __name__ == '__main__':
    PARAMS = __init__()
    print(PARAMS['today'])

    misc.print_configuration(PARAMS)

    PARAMS['opDir_SPT'] = PARAMS['opDir'] + '/SPT/'
    if not os.path.exists(PARAMS['opDir_SPT']):
        os.makedirs(PARAMS['opDir_SPT'])
    
    compute_SPT_matrices(PARAMS, PARAMS['train_files']['wav'])
    print('train files SPT done')
    compute_SPT_matrices(PARAMS, PARAMS['val_files']['wav'])
    print('val files SPT done')
    compute_SPT_matrices(PARAMS, PARAMS['test_files']['wav'])
    print('test files SPT done')
    
    # sys.exit(0)


    PARAMS['opDir_temp'] = PARAMS['opDir'] + '/__temp/'
    if not os.path.exists(PARAMS['opDir_temp']):
        os.makedirs(PARAMS['opDir_temp'])

    PARAMS['gmmPath_aspt'] = PARAMS['opDir'] + '/gmm_aspt/'
    if not os.path.exists(PARAMS['gmmPath_aspt']):
        os.makedirs(PARAMS['gmmPath_aspt'])
    
    PARAMS['gmmPath_lspt'] = PARAMS['opDir'] + '/gmm_lspt/'
    if not os.path.exists(PARAMS['gmmPath_lspt']):
        os.makedirs(PARAMS['gmmPath_lspt'])
            
    for feat_type in ['aspt', 'lspt']:
        for clNum in PARAMS['classes'].keys():
            classname = PARAMS['classes'][clNum]
            print('Computing features for ', classname)            
            train_files = PARAMS['train_files']['wav']
            seqCount = 0
            for seqNum in range(PARAMS['nTopPeaks']):
                print(f'{feat_type} Sequence{seqNum}: {classname} GMM computation')
                optFileGMM = PARAMS['gmmPath_'+feat_type] + '/Cl' + str(clNum) + '_seq' + str(seqNum) + '_train_data_gmm.pkl'
                if not os.path.exists(optFileGMM):
                    tempFile = PARAMS['opDir_temp'] + '/train_cl' + str(clNum) + '_seq' + str(seqNum) + '_' + feat_type + '.npz'
                    if not os.path.exists(tempFile):
                        tr_data_seq, tr_file_mark, tr_data_file_mark = load_files_cbow_features(PARAMS, train_files, seqNum, feat_type, classname)
                        np.savez(tempFile, data_seq=tr_data_seq.astype(np.float32), file_mark=tr_file_mark.astype(np.float32), data_file_mark=tr_data_file_mark)
                    else:
                        tr_data_seq = np.load(tempFile)['data_seq']
                        tr_file_mark = np.load(tempFile)['file_mark']
                        tr_data_file_mark = np.load(tempFile, allow_pickle=True)['data_file_mark']
                        #print('                  Train seq data loaded')
                    print('\t\tTraining files read ', np.shape(tr_data_seq))    
                    learn_seq_gmm(PARAMS, tr_data_seq, optFileGMM)
                    print(f'{feat_type} cl{clNum} seq{seqNum} GMM trained')
                else:
                    print(f'{feat_type} GMM exits for cl={clNum} seq={seqNum}')

    PARAMS['opDirMSD_ASPT_LSPT'] = PARAMS['opDir'] + '/MSD-ASPT-LSPT/'
    if not os.path.exists(PARAMS['opDirMSD_ASPT_LSPT']):
        os.makedirs(PARAMS['opDirMSD_ASPT_LSPT'])

    PARAMS['opDirCBoW_ASPT_LSPT'] = PARAMS['opDir'] + '/CBoW-ASPT-LSPT/'
    if not os.path.exists(PARAMS['opDirCBoW_ASPT_LSPT']):
        os.makedirs(PARAMS['opDirCBoW_ASPT_LSPT'])

    PARAMS['opDirStats_ASPT_LSPT'] = PARAMS['opDir'] + '/Stats-ASPT-LSPT/'
    if not os.path.exists(PARAMS['opDirStats_ASPT_LSPT']):
        os.makedirs(PARAMS['opDirStats_ASPT_LSPT'])
        
    compute_features(PARAMS, PARAMS['train_files']['wav'], 'train')
    compute_features(PARAMS, PARAMS['val_files']['wav'], 'val')
    compute_features(PARAMS, PARAMS['test_files']['wav'], 'test')
    