#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:01:03 2019
Updated on Tue Nov 16 14:09:00 2021

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import os
import lib.misc as misc
from sklearn import mixture
import pickle
import sys



'''
This function learns GMMs by taking all data of any i^th sequence at a time. 
Finally, k GMMs are learned, where k is the number of sequences in the the SPS matrix
''' 
def learn_seq_gmm(PARAMS, seq_data, optFileGMM):
    flattened_seq_data = np.transpose(np.array(seq_data.flatten(), ndmin=2))
    gmm = mixture.GaussianMixture(n_components=PARAMS['numMix'], covariance_type='full')

    gmm.fit(flattened_seq_data)
    if PARAMS['save_flag']:
        print('Saving GMM model as ', optFileGMM)
        with open(optFileGMM, 'wb') as file:
            pickle.dump(gmm, file)



'''
This function computes posterior probability features
''' 
def get_gmm_posterior_features(PARAMS, seqNum, seq_data, file_mark, data_type):
    data_lkhd = np.empty(shape=(np.size(file_mark),len(PARAMS['classes'])*PARAMS['numMix']))
    
    '''
    Generating the likelihood features from the learned GMMs
    '''
    print('Generating likelihood features')
    file_mark_start = 0
    seq_data = np.array(seq_data, ndmin=2)
    print('Shape of seq_data: ', np.shape(seq_data), np.shape(data_lkhd))
    
    for i in range(len(file_mark)):
        flattened_test_data = np.transpose(np.array(seq_data[0,file_mark_start:file_mark[i]], ndmin=2))
        if not np.size(flattened_test_data)>1:
            continue
    
        for clNum in PARAMS['classes'].keys():
            gmm = None
            '''
            Load already learned GMM
            '''
            optFileGMM = PARAMS['gmmPath'] + '/Cl' + str(clNum) + '_seq' + str(seqNum) + '_train_data_gmm.pkl'
            with open(optFileGMM, 'rb') as file:  
                gmm = pickle.load(file)

            proba = gmm.predict_proba(flattened_test_data)
            mean_idx = np.ndarray.argsort(np.squeeze(gmm.means_))
            proba = proba[:, mean_idx]
            proba_fv = np.mean(proba, axis=0)
    
            data_lkhd[i,clNum*PARAMS['numMix']:(clNum+1)*PARAMS['numMix']] = proba_fv
        
        file_mark_start = file_mark[i]
        
    print('Likelihood features computed')

    return data_lkhd




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



'''
This function reads in the files listed and returns their contents as a single
2D array
'''
def load_files_cbow_features(data_path, files, seqNum, feat_type):
    data = np.empty([],dtype=float)
    file_mark = np.empty([],dtype=float)
    data_file_mark = {}
    
    fileCount = 0
    data_marking_start = 0
    data_marking = 0
    for fl in files:
        # fName = data_path + '/' + fl
        # SPS_matrix = np.load(fName, allow_pickle=True)
        if feat_type=='CBoW-ASPT':
            SPS_matrix = misc.load_obj(data_path, fl.split('.')[0])['val']
        elif feat_type=='CBoW-LSPT':
            SPS_matrix = misc.load_obj(data_path, fl.split('.')[0])['loc']

        row = SPS_matrix[:, :, seqNum]
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
        
        fileCount += 1
    file_mark = np.cumsum(file_mark).astype(int)
    print('Reading files ... ',fileCount, '/', len(files))
    
    print('Data loaded: ', np.shape(data), np.shape(file_mark))
    return data, file_mark, data_file_mark




def __init__():
    PARAMS = {
            # 'folder': './features/musan/1000ms_10ms_5ms_10PT/', # Laptop
            'folder': '/home/mrinmoy/Documents/PhD_Work/Features/MTGC_SMO/musan/1000ms_10ms_5ms_10PT/', # LabPC
            'numMix': 5,
            'num_seq': 10,
            'save_flag': True,
            'classes':{0:'music', 1:'speech'},
            'CV_folds': 3,
            'fold': 0,
            'feat_type': 'CBoW-ASPT',
            }

    PARAMS['dataset_name'] = 'musan' # list(filter(None,PARAMS['folder'].split('/')))[-1]
    PARAMS['CV_path_train'] = './cross_validation_info/' + PARAMS['dataset_name'] + '/'    
    if not os.path.exists(PARAMS['CV_path_train']+'/cv_file_list.pkl'):
        print('Fold division of files for cross-validation not done!')
        sys.exit(0)
    PARAMS['cv_file_list'] = misc.load_obj(PARAMS['CV_path_train'], 'cv_file_list')
    
    return PARAMS



if __name__ == '__main__':
    PARAMS = __init__()

    ''' This function generates CBoW feature from the Peak trace matrices of 
    the classes by learning SEPARATE GMMs and extracting posterior 
    probabilities from them '''
    
    for PARAMS['fold'] in range(PARAMS['CV_folds']):
        PARAMS['train_files'], PARAMS['test_files'] = get_train_test_files(PARAMS['cv_file_list'], PARAMS['CV_folds'], PARAMS['fold'])
        PARAMS['output_folder'] = PARAMS['folder'] + '/' + PARAMS['feat_type'] + '_mix'+ str(PARAMS['numMix'])
        PARAMS['opDir'] = PARAMS['output_folder'] + '/fold' + str(PARAMS['fold']) + '/'
        if not os.path.exists(PARAMS['opDir']):
            os.makedirs(PARAMS['opDir'])
            
        PARAMS['tempFolder'] = PARAMS['output_folder'] + '/__temp/fold' + str(PARAMS['fold']) + '/'
        if not os.path.exists(PARAMS['tempFolder']):
            os.makedirs(PARAMS['tempFolder'])
        PARAMS['gmmPath'] = PARAMS['output_folder'] + '/__GMMs/fold' + str(PARAMS['fold']) + '/'
        if not os.path.exists(PARAMS['gmmPath']):
            os.makedirs(PARAMS['gmmPath'])

        for clNum in PARAMS['classes'].keys():
            classname = PARAMS['classes'][clNum]
            data_path = PARAMS['folder'] + '/SPT/' + classname + '/' 	#path where the peak trace matrices are
            print('Computing features for ', classname)
            
            train_files = PARAMS['train_files'][classname]

            seqCount = 0
            for seqNum in range(PARAMS['num_seq']):
                ''' Reading in the files and storing each sequence as a 
                separate feature vector appended with its corresponding file 
                mark information '''
                print('\n.................................................... Seq:', seqNum, ' : ', PARAMS['classes'][clNum])
                
                tempFile = PARAMS['tempFolder'] + '/train_cl' + str(clNum) + '_seq' + str(seqNum) + '.npz'
                if not os.path.exists(tempFile):
                    tr_data_seq, tr_file_mark, tr_data_file_mark = load_files_cbow_features(data_path, train_files, seqNum, PARAMS['feat_type'])
                    np.savez(tempFile, data_seq=tr_data_seq, file_mark=tr_file_mark, data_file_mark=tr_data_file_mark)
                else:
                    tr_data_seq = np.load(tempFile)['data_seq']
                    tr_file_mark = np.load(tempFile)['file_mark']
                    tr_data_file_mark = np.load(tempFile, allow_pickle=True)['data_file_mark']
                    #print('                  Train seq data loaded')
                print('\t\tTraining files read ', np.shape(tr_data_seq))
    

                '''
                Learning GMMs on training data
                '''
                optFileGMM = PARAMS['gmmPath'] + 'Cl' + str(clNum) + '_seq' + str(seqNum) + '_train_data_gmm.pkl'
                if not os.path.exists(optFileGMM):
                    learn_seq_gmm(PARAMS, tr_data_seq, optFileGMM)
                else:
                    print('GMM exits for cl=', clNum, ' seq=', seqNum)


        
        for clNum in range(len(PARAMS['classes'])):
            classname = PARAMS['classes'][clNum]
            data_path = PARAMS['folder'] + '/SPT/' + classname + '/' 	#path where the Peak trace matrices are
            print('Computing features for ', classname)
            
            tr_data_lkhd = np.empty([])
            ts_data_lkhd = np.empty([])

            tr_data_lkhd_merged = np.empty([])
            ts_data_lkhd_merged = np.empty([])
            
            train_files = PARAMS['train_files'][classname]
            test_files = PARAMS['test_files'][classname]
            
            seqCount = 0
            for seqNum in range(PARAMS['num_seq']):
                ''' Reading in the files and storing each sequence as a separate 
                feature vector appended with its corresponding file mark 
                information '''
                print('\n.................................................... Seq:', seqNum, ' : ', PARAMS['classes'][clNum])
                
                # -----------------------------------------------------------------
                tempFile = PARAMS['tempFolder'] + '/train_cl' + str(clNum) + '_seq' + str(seqNum) + '.npz'
                if not os.path.exists(tempFile):
                    tr_data_seq, tr_file_mark, tr_data_file_mark = load_files_cbow_features(data_path, train_files, seqNum, PARAMS['feat_type'])
                    np.savez(tempFile, data_seq=tr_data_seq, file_mark=tr_file_mark, data_file_mark=tr_data_file_mark)
                else:
                    tr_data_seq = np.load(tempFile)['data_seq']
                    tr_file_mark = np.load(tempFile)['file_mark']
                    tr_data_file_mark = np.load(tempFile, allow_pickle=True)['data_file_mark']
                    #print('                  Train seq data loaded')
                print('\t\tTraining files read ', np.shape(tr_data_seq))
                # -----------------------------------------------------------------
        
                # -----------------------------------------------------------------
                tempFile = PARAMS['tempFolder'] + '/test_cl' + str(clNum) + '_seq' + str(seqNum) + '.npz'
                if not os.path.exists(tempFile):
                    ts_data_seq, ts_file_mark, ts_data_file_mark = load_files_cbow_features(data_path, test_files, seqNum, PARAMS['feat_type'])
                    np.savez(tempFile, data_seq=ts_data_seq, file_mark=ts_file_mark, data_file_mark=ts_data_file_mark)
                else:
                    ts_data_seq = np.load(tempFile)['data_seq']
                    ts_file_mark = np.load(tempFile)['file_mark']
                    ts_data_file_mark = np.load(tempFile, allow_pickle=True)['data_file_mark']
                    #print('                  Test seq data loaded')
                print('\t\testing files read ', np.shape(ts_data_seq))                
                # -----------------------------------------------------------------
                
                ''' Generating GMM likelihood features for each sequence '''
                print('Computing GMM likelihood features ...')
                tr_data_lkhd = get_gmm_posterior_features(PARAMS, seqNum, tr_data_seq, tr_file_mark, 'train_data')
                print('\t\t\tTraining set done ', np.shape(tr_data_lkhd))
    
                ts_data_lkhd = get_gmm_posterior_features(PARAMS, seqNum, ts_data_seq, ts_file_mark, 'test_data')
                print('\t\t\tTesting set done ', np.shape(ts_data_lkhd))
                # -----------------------------------------------------------------
                
                
                if np.size(tr_data_lkhd_merged)<=1:
                    tr_data_lkhd_merged = tr_data_lkhd
                    ts_data_lkhd_merged = ts_data_lkhd
                else:
                    tr_data_lkhd_merged = np.append(tr_data_lkhd_merged, tr_data_lkhd, 1)
                    ts_data_lkhd_merged = np.append(ts_data_lkhd_merged, ts_data_lkhd, 1)
                    
                seqCount += 1
                if np.sum(np.isnan(tr_data_lkhd))>0:
                    print('\n\n\nNaNs seqNum=',seqNum,'\n\n\n')
                    
            
            ''' To save the CBoW interval features file wise so that they can 
            be merged later on '''
            opDirFold = PARAMS['opDir'] + '/' + classname + '/'
            if not os.path.exists(opDirFold):
                os.makedirs(opDirFold)
                
            # Dict gets stored as a numpy.ndarray object, so .item() is required
            print('tr_data_file_mark: ', tr_data_file_mark)
            if type(tr_data_file_mark)!=dict:
                tr_data_file_mark = tr_data_file_mark.item()
            for key in tr_data_file_mark.keys():
                featFile = opDirFold + key.split('.')[0]
                idx = tr_data_file_mark[key][0]
                fileData = tr_data_lkhd_merged[idx[0]:idx[1],:]
                np.save(featFile, fileData)
                
            print('ts_data_file_mark: ', ts_data_file_mark)
            if type(ts_data_file_mark)!=dict:
                ts_data_file_mark = ts_data_file_mark.item()
            for key in ts_data_file_mark.keys():
                featFile = opDirFold + key.split('.')[0]
                idx = ts_data_file_mark[key][0]
                fileData = ts_data_lkhd_merged[idx[0]:idx[1],:]
                np.save(featFile, fileData)
            ''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
        print('GMM likelihood features saved.')

    print('\n\n\nGenerated GMM likelihood features for ', PARAMS['num_seq'],' sequences and ', PARAMS['numMix'],' mixtures.\n')
    
