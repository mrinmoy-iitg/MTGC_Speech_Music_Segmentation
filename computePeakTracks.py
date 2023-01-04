#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 20:38:10 2019

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import lib.spectral_peak_tracking as SPT
import datetime
import os
import librosa.display
import random
import time
#import lib.audio_processing as aproc
import json
import lib.misc as misc





def __init__():
    PARAMS = {
            'Tw': 10, # Frame size in miliseconds
            'Ts': 5, # Frame shift in miliseconds
            'nTopPeaks': 10, # Number of Prominent peaks to be considered for SPS computation
            'silThresh': 0.0,  # Silence Threshold
            'intervalSize': 1000, # In miliseconds
            'intervalShift': 1000, # In miliseconds
            # 'audio_path': '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/musan/', # Laptop
            # 'output_path': './features/', # Laptop
            'audio_path': '/media/mrinmoy/Windows_Volume/PhD_Work/data/musan', # LabPC 
            'output_path': '/home/mrinmoy/Documents/PhD_Work/Features/MTGC_SMO/', # LabPC
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            'classes':{0:'music', 1:'speech'},
            }

    PARAMS['dataset_name'] = list(filter(None,PARAMS['audio_path'].split('/')))[-1]
    PARAMS['opDir'] = PARAMS['output_path'] + '/' + PARAMS['dataset_name']+'/'+ str(PARAMS['intervalSize']) + 'ms_' + str(PARAMS['Tw'])+'ms_'+str(PARAMS['Ts']).replace('.','-')+'ms_'+ str(PARAMS['nTopPeaks']) +'PT_' + PARAMS['today']
    if not os.path.exists(PARAMS['opDir']):
        os.makedirs(PARAMS['opDir'])

    with open(PARAMS['opDir'] + '/configuration.txt', 'a+', encoding = 'utf-8') as file:
        file.write(json.dumps(PARAMS))
    
    return PARAMS




if __name__ == '__main__':
    PARAMS = __init__()
    print(PARAMS['today'])

    opDir_SPT = PARAMS['opDir'] + '/SPT/'
    if not os.path.exists(opDir_SPT):
        os.makedirs(opDir_SPT)
        
    opDirMSD_ASPT_LSPT = PARAMS['opDir'] + '/MSD-ASPT-LSPT/'
    if not os.path.exists(opDirMSD_ASPT_LSPT):
        os.makedirs(opDirMSD_ASPT_LSPT)

    clNum = -1
    for clNum in PARAMS['classes'].keys():
        fold = PARAMS['classes'][clNum]
        
        path = PARAMS['audio_path'] + '/' + fold + '/' 	#path where the audio is
        print('\n\n',path)
        files = librosa.util.find_files(path, ext=['wav'])
        numFiles = np.size(files)
        
        randFileIdx = list(range(np.size(files)))
        random.shuffle(randFileIdx)
        
        opDir_SPT_Fold = opDir_SPT + fold + '/'
        if not os.path.exists(opDir_SPT_Fold):
            os.makedirs(opDir_SPT_Fold)
            
        opDirMSD_ASPT_LSPT_Fold = opDirMSD_ASPT_LSPT + fold + '/'
        if not os.path.exists(opDirMSD_ASPT_LSPT_Fold):
            os.makedirs(opDirMSD_ASPT_LSPT_Fold)

        fileCount = 0 #to see in the terminal how many files have been loaded
        for f in range(numFiles):
            audio = files[randFileIdx[f]]
            fileCount += 1
            print('\n', fileCount, ' Audio file: ', audio, fold)
            
            check_file = opDir_SPT_Fold + audio.split('/')[-1].split('.')[0] + '.pkl'
            print('Check file: ', check_file)
            if os.path.exists(check_file):
                print('Feature file already exists!!!\n\n\n')
                continue
                        
            Xin, fs = librosa.core.load(audio, mono=True, sr=None)
            print('Sampling rate=', fs)
            
            # # Removing the silences
            # Xin_silrem, silPos = aproc.removeSilence(Xin=Xin, Tw=PARAMS['Tw'], Ts=PARAMS['Ts'], fs=fs, threshold=PARAMS['silThresh'])
            # print('File duration: ', np.round(len(Xin_silrem)/fs,2),' (', np.round(len(Xin)/fs,2), ')')
            # if np.size(Xin_silrem)<=1:
            #     continue;
            # Xin = Xin_silrem
            # if len(Xin)/fs < 1:
            #     continue

            frameSize = int((PARAMS['Tw']*fs)/1000) # Frame size in number of samples
            frameShift = int((PARAMS['Ts']*fs)/1000) # Frame shift in number of samples

            '''
            Creating the corresponding SPT matrix
            '''
            start = time.process_time()
            
            LSPT, ASPT, stft, peak_repeat_count = SPT.computeSPT(y=Xin, sr=fs, frame_length=frameSize, hop_length=frameShift, req_pks=PARAMS['nTopPeaks'])
            
            # LSPT, ASPT, stft, peak_repeat_count = SPT.computeSPT_Parallel(y=Xin, sr=fs, frame_length=frameSize, hop_length=frameShift, req_pks=PARAMS['nTopPeaks'])
            
            print('Time for SPT: ',np.round(time.process_time()-start,2),'s')
            
            numSamplesInInterval = int(PARAMS['intervalSize']*fs/1000)
            numSamplesShiftPerInterval = int(PARAMS['intervalShift']*fs/1000)
            numFv = int(np.floor((len(Xin)-numSamplesInInterval)/numSamplesShiftPerInterval))+1
            nFrames = np.shape(ASPT)[0]

            peak_repeat_statistics_file = PARAMS['opDir'] + '/' + fold + '_peak_repeat_statistics.csv'
            fid = open(peak_repeat_statistics_file, 'a+', encoding='utf8')
            fid.write(audio + '\t' + str(peak_repeat_count) + '\t' + str(nFrames) + '\n')
            fid.close()
            
            if numFv==0:
                continue
            nmFrmPerInterval = int(np.floor((numSamplesInInterval-frameSize)/frameShift))+1
            nmFrmPerIntervalShift = int(np.floor((numSamplesShiftPerInterval-frameSize)/frameShift))+1
            print('Signal: ', len(Xin), numSamplesInInterval, numFv, nFrames, nmFrmPerInterval, np.shape(ASPT))
            
            frmStart = 0
            frmEnd = 0
            fv_val = np.array
            fv_loc = np.array
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
            misc.save_obj(SPT_data, opDir_SPT_Fold, audio.split('/')[-1].split('.')[0])
            print('APT: ', np.shape(fv_val), ' LPT: ', np.shape(fv_loc), ' Sampling rate: ', fs)

            start = time.process_time()
            MSD_LSPT = SPT.computeSPT_MeanStd(numFv, nmFrmPerInterval, nmFrmPerIntervalShift, nFrames, LSPT)
            MSD_LSPT = np.array(MSD_LSPT, ndmin=2)
            print('MSD_LSPT: ', np.shape(MSD_LSPT))

            MSD_ASPT = SPT.computeSPT_MeanStd(numFv, nmFrmPerInterval, nmFrmPerIntervalShift, nFrames, ASPT)
            MSD_ASPT = np.array(MSD_ASPT, ndmin=2)
            print('MSD_ASPT: ', np.shape(MSD_ASPT))

            MSD_ASPT_LSPT = np.append(MSD_ASPT, MSD_LSPT, 1)
            MSD_ASPT_LSPT = np.array(MSD_ASPT_LSPT, ndmin=2)
            print('MSD_ASPT_LSPT: ', np.shape(MSD_ASPT_LSPT))
            fileName = opDirMSD_ASPT_LSPT_Fold + audio.split('/')[-1].split('.')[0] + '.npy'
            np.save(fileName, MSD_ASPT_LSPT)

            print('Time for MSD Features: ',np.round(time.process_time() - start,2),'s')
