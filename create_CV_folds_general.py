#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 19:09:15 2021

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import os
import librosa
import numpy as np
import lib.misc as misc
import csv


def duration_format(N):
    '''
    Conversion of number of seconds into HH-mm-ss format
    
    Parameters
    ----------
    N : int
        Number of seconds.

    Returns
    -------
    H : int
        Hours.
    M : int
        Minutes.
    S : int
        Seconds.

    Created: 24 July, 2021
    '''
    H = int(N/3600)
    N = N%3600
    M = int(N/60)
    S = N%60
    return H,M,S


def calculate_dataset_size(folder, classes):
    '''
    This function calculates the size of the dataset indicated by the path in
    "folder"

    Parameters
    ----------
    folder : str
        Path to dataset.
    classes : dict
        Key-value pairs of class-labels and names.

    Returns
    -------
    duration : dict
        Key-value pairs of classnames and their durations.
    filewise_duration : dict
        Key-value pairs of filenames and their durations.

    Created: 24 July, 2021
    '''
    
    duration = {classes[key]:0 for key in classes}
    filewise_duration = {classes[key]:{} for key in classes}
    for classname in duration:
        class_data_path = folder + '/' + classname + '/'
        files = [fl.split('/')[-1] for fl in librosa.util.find_files(class_data_path, ext=['wav'])]
        count = 0
        progress_mark = ['-', '\\', '|', '/']
        for fl in files:
            fName = class_data_path + fl
            Xin, fs = librosa.core.load(fName, mono=True, sr=None)
            file_duration = len(Xin)/fs
            duration[classname] += file_duration
            filewise_duration[classname][fl] = file_duration
            H,M,S = duration_format(int(duration[classname]))
            print(classname, progress_mark[count%4], H, 'hr', M, 'min', S, 'sec', end='\r', flush=True)
            count += 1
        print('\n')
    return duration, filewise_duration




def get_annotations(folder):
    annotations = {}
    genre_list = {}
    genre_id = 0
    with open(folder+'/Annotations.csv') as annot_file:
        annotreader = csv.reader(annot_file, delimiter=',', quotechar='|')        
        row_count = 0
        for row in annotreader:
            if row==[]:
                continue
            if row_count==0:
                row_count += 1
                continue
            annotations[row[0]] = {'movie_title':row[2], 'genre':row[1].split('|')}
            for genre in row[1].split('|'):
                if not genre in genre_list.keys():
                    genre_list[genre] = genre_id
                    genre_id += 1
            row_count += 1
    return annotations, genre_list




def create_CV_folds(folder, dataset_name, classes, CV, total_duration, filewise_duration):
    '''
    This function divides the audio files in the speech and music classes of 
    the dataset into non-overlapping folds.

    Parameters
    ----------
    folder : str
        Path to the dataset.
    dataset_name : str
        Name of the dataset.
    classes : dict
        A dict containing class labels and names in key-value pairs.
    CV : int
        Number of cross-validation folds to be created.
    total_duration : dict
        A dict containing the total durations of speech and music classes as 
        key-value pairs.
    filewise_duration : dict
        A dict containing the filenames and their respective durations as 
        key-value pairs for speech and music classes.

    Returns
    -------
    cv_file_list : TYPE
        Key-value pairs of classes and the list of files divided into 
        non-overlapping folds.
    
    Created: 24 July, 2021
    '''
    annotations, genre_list = get_annotations(folder)

    cv_file_list = {
        'CV_folds': CV, 
        'dataset_name': dataset_name, 
        classes[0]:{'fold'+str(i):[] for i in range(CV)}
        }
    annot_count = np.zeros((CV, len(genre_list)))
    for fName in annotations.keys():
        if os.path.exists(folder + '/' + classes[0] + '/' + fName + '.wav'):
            genres = annotations[fName]['genre']
            print(fName, genres, np.shape(annot_count))
            min_fold_id = []
            for genre_i in genres:
                gid = np.squeeze(np.where([key==genre_i for key in genre_list.keys()]))
                min_fold_id.append(np.argmin(annot_count[:,gid]))
            select_fold = np.min(min_fold_id)
            for genre_i in genres:
                gid = np.squeeze(np.where([key==genre_i for key in genre_list.keys()]))
                annot_count[select_fold,gid] += 1
            cv_file_list[classes[0]]['fold'+str(select_fold)].append(fName+'.wav')
        
    for clNum in classes.keys():
        path = folder + '/' + classes[clNum] + '/'
        files = [fl.split('/')[-1] for fl in librosa.util.find_files(path, ext=['wav'])]
        np.random.shuffle(files)            
        for cv_num in range(CV):
            fold_duration = 0
            # print(cv_file_list[classes[clNum]].keys())
            for fl in cv_file_list[classes[clNum]]['fold'+str(cv_num)]:
                fold_duration += filewise_duration[classes[clNum]][fl]

    cv_file_list['filewise_duration'] = filewise_duration
    cv_file_list['total_duration'] = total_duration
    for key in cv_file_list['total_duration'].keys():
        cv_file_list['total_duration'][key] /= 3600 # in Hours

    print('total_duration: ', total_duration)
    dataset_size = 0 # in Hours
    for classname in cv_file_list['total_duration'].keys():
        dataset_size += cv_file_list['total_duration'][classname]
    print('Dataset size: ', dataset_size, 'Hrs', total_duration)
    cv_file_list['dataset_size'] = dataset_size
    
    return cv_file_list, genre_list, annot_count



def print_cv_info(cv_file_list, classes, opDir, CV):
    '''
    Print the DESCRIPTIONresult of cross-validation fold distribution of files generated 
    for the dataset

    Parameters
    ----------
    cv_file_list : dict
        Class-wise key-value pairs of files in different cross-validation 
        folds.
    opDir : str
        Path to store the result files.
    CV : int
        Number of cross-validation folds.

    Returns
    -------
    None.

    Created: 24 July, 2021
    '''
    fid = open(opDir+'/details.txt', 'w+', encoding='utf8')
    for key in cv_file_list.keys():
        fid.write(key + ': ' + str(cv_file_list[key]) +'\n\n\n')
    fid.close()
    
    for fold in range(CV):
        fid = open(opDir+'/fold' + str(fold) + '.csv', 'w+', encoding='utf8')
        files = cv_file_list[classes[0]]['fold'+str(fold)]
        # print('fold', fold, files)
        for fl in files:
            fid.write(fl+'\n')
        fid.close()    



if __name__ == '__main__':
    folder = '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/Moviescope/' # LabPC
    classes = {0:'wav'}
    CV = 3
    dataset_name = list(filter(None,folder.split('/')))[-1]
    opDir = './cross_validation_info/' + dataset_name + '/'
    if not os.path.exists(opDir):
        os.makedirs(opDir)
            
    if not os.path.exists(opDir+'/Dataset_Duration.pkl'):
        total_duration, filewise_duration = calculate_dataset_size(folder, classes)
        misc.save_obj({'total_duration':total_duration, 'filewise_duration':filewise_duration}, opDir, 'Dataset_Duration')
    else:
        total_duration = misc.load_obj(opDir, 'Dataset_Duration')['total_duration']
        filewise_duration = misc.load_obj(opDir, 'Dataset_Duration')['filewise_duration']    
    
    if not os.path.exists(opDir+'/cv_file_list.pkl'):
        cv_file_list, genre_list, annot_count = create_CV_folds(folder, dataset_name, classes, CV, total_duration, filewise_duration)
        misc.save_obj(cv_file_list, opDir, 'cv_file_list')
        misc.save_obj(genre_list, opDir, 'genre_list')
        misc.save_obj(annot_count, opDir, 'annot_count')
        print('CV folds created')
    else:
        cv_file_list = misc.load_obj(opDir, 'cv_file_list')
        genre_list = misc.load_obj(opDir, 'genre_list')
        annot_count = misc.load_obj(opDir, 'annot_count')
        print('CV folds loaded')
    
    # print('annot_count: ', annot_count)
    # print('cv_file_list: ', cv_file_list)
    print_cv_info(cv_file_list, classes, opDir, CV)
