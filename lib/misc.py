#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:55:09 2019

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""
import os
import numpy as np
import pickle
from sklearn import preprocessing
import json
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from scipy.spatial import distance
import csv




def save_obj(obj, folder, name):
    with open(folder+'/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)




def load_obj(folder, name):
    with open(folder+'/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)




def print_configuration(PARAMS):
    opFile = PARAMS['opDir'] + '/Configuration.csv'
    fid = open(opFile, 'a+', encoding = 'utf-8')
    PARAM_keys = [key for key in PARAMS.keys()]
    for i in range(len(PARAM_keys)):
        if PARAM_keys[i]=='GPU_session':
            continue
        # print('PARAMS key: ', PARAM_keys[i])
        try:
            fid.write(PARAM_keys[i] + '\t')
            fid.write(json.dumps(PARAMS[PARAM_keys[i]]))
            fid.write('\n')
        except:
            fid.write(PARAM_keys[i] + '\tERROR\n')
            
    fid.close()
    
    


def getPerformance(PtdLabels, GroundTruths, labels):
    ConfMat = confusion_matrix(y_true=GroundTruths, y_pred=PtdLabels)
    precision, recall, fscore, support = precision_recall_fscore_support(y_true=GroundTruths, y_pred=PtdLabels, beta=1.0, average=None, labels=labels)
    precision = np.round(precision,4)
    recall = np.round(recall,4)
    fscore = np.round(fscore,4)
    fscore = np.append(fscore, np.mean(fscore))

    return ConfMat, precision, recall, fscore




def print_results(PARAMS, suffix, **kwargs):
    opFile = PARAMS['opDir'] + '/Performance' + suffix + '.csv'

    linecount = 0
    if os.path.exists(opFile):
        with open(opFile, 'r', encoding='utf8') as fid:
            for line in fid:
                linecount += 1    
    
    fid = open(opFile, 'a+', encoding = 'utf-8')
    heading = 'fold'
    values = str(PARAMS['fold'])
    for i in range(len(kwargs)):
        heading = heading + '\t' + np.squeeze(kwargs[str(i)]).tolist().split(':')[0]
        values = values + '\t' + np.squeeze(kwargs[str(i)]).tolist().split(':')[1]

    if linecount==0:
        fid.write(heading + '\n' + values + '\n')
    else:
        fid.write(values + '\n')
        
    fid.close()




def print_model_summary(arch_file, model):
    stringlist = ['']
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    with open(arch_file, 'w+', encoding='utf8') as f:
        f.write(short_model_summary)



def scale_data(train_data, val_data, test_data, **kwargs):
    All_train_data = train_data
    All_train_data = np.append(All_train_data, val_data, 0)
    if np.shape(All_train_data)[1]>np.shape(All_train_data)[0]:
        All_train_data = All_train_data.T
    print('Scale data func: All_train_data=', np.shape(All_train_data))
    if 'std_scaler' in kwargs:
        std_scaler = kwargs['std_scaler']
    else:
        std_scaler = preprocessing.StandardScaler().fit(All_train_data)

    train_data_scaled = std_scaler.transform(train_data)
    val_data_scaled = std_scaler.transform(val_data)
    test_data_scaled = std_scaler.transform(test_data)

    return train_data_scaled, val_data_scaled, test_data_scaled, std_scaler




def load_data_from_files(PARAMS, file_list):
    label = np.empty([])
    data = np.empty([])
    
    for clNum in PARAMS['classes'].keys():
        files = file_list[PARAMS['classes'][clNum]]
        for fl in files:
            fName = PARAMS['folder'] + '/' + PARAMS['classes'][clNum] + '/' + fl.split('.')[0] + '.npy'
            FV = np.load(fName, allow_pickle=True)
            if np.size(data)<=1:
                data = FV
                label = np.array([clNum]*np.shape(FV)[0])
            else:
                data = np.append(data, FV, axis=0)
                label = np.append(label, np.array([clNum]*np.shape(FV)[0]))
            # print('load_data_from_files: ', np.shape(data), np.shape(label), np.unique(label), clNum, PARAMS['classes'][clNum])
    return data, label


   
def get_train_val_test_data(PARAMS):    
    train_files = {}
    val_files = {}
    for classname in  PARAMS['train_files'].keys():
        files = PARAMS['train_files'][classname]
        np.random.shuffle(files)
        nTrain = int(len(files)*0.7)
        train_files[classname] = files[:nTrain]
        val_files[classname] = files[nTrain:]
        print(classname, nTrain, len(files)-nTrain)

    print('Loading training data ')
    train_data, train_label = load_data_from_files(PARAMS, train_files)
        
    print('Loading validation data ')
    val_data, val_label = load_data_from_files(PARAMS, val_files)

    print('Loading testing data ')
    test_data, test_label = load_data_from_files(PARAMS, PARAMS['test_files'])
    
    All_Data_Splits = {
        'train_data': train_data,
        'train_label': train_label,
        'val_data': val_data,
        'val_label': val_label,
        'test_data': test_data,
        'test_label': test_label,
        }
    
    if PARAMS['save_flag']:
        save_obj(All_Data_Splits, PARAMS['opDir'], 'All_Data_Splits_fold'+str(PARAMS['fold']))

    print('Train data shape: ', train_data.shape, train_label.shape, np.unique(train_label))
    print('Val data shape: ', val_data.shape, val_label.shape, np.unique(val_label))
    print('Test data shape: ', test_data.shape, test_label.shape, np.unique(test_label))

    return All_Data_Splits





'''
Read all data from folder and split it into train, validation and test data
'''
def get_data(PARAMS):
    savedData = PARAMS['opDir']+'/All_Data_Splits_fold'+str(PARAMS['fold'])+'.pkl'
    print('Saved data: ', savedData)
    if not os.path.exists(savedData):
        print('Training data not available !!!')
        All_Data_Splits = get_train_val_test_data(PARAMS)   
    else:
        All_Data_Splits = load_obj(PARAMS['opDir'], 'All_Data_Splits_fold'+str(PARAMS['fold']))            

    print(np.shape(All_Data_Splits['train_data']), np.shape(All_Data_Splits['val_data']), np.shape(All_Data_Splits['test_data']))
   
    '''
    Scaling the data
    '''
    if not os.path.exists(PARAMS['opDir']+'/std_scaler_fold'+str(PARAMS['fold'])+'.pkl'):
        train_data, val_data, test_data, std_scaler = scale_data(All_Data_Splits['train_data'], All_Data_Splits['val_data'], All_Data_Splits['test_data'])
        save_obj(std_scaler, PARAMS['opDir'], 'std_scaler_fold'+str(PARAMS['fold']))
    else:
        std_scaler = load_obj(PARAMS['opDir'], 'std_scaler_fold'+str(PARAMS['fold']))
        train_data, val_data, test_data, std_scaler = scale_data(All_Data_Splits['train_data'], All_Data_Splits['val_data'], All_Data_Splits['test_data'], std_scaler=std_scaler)

    print('Data shape: ', np.shape(train_data), np.shape(val_data), np.shape(test_data))
    data_dict = {
            'train_data': train_data,
            'train_label': All_Data_Splits['train_label'],
            'val_data': val_data,
            'val_label': All_Data_Splits['val_label'],
            'test_data': test_data,
            'test_label': All_Data_Splits['test_label'],
            }
    return data_dict



def compute_bic(kmeans, X):
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * np.sum([np.sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return BIC




def get_annotations(path, possible_genres):
    annotations = {}
    genre_list = {}
    genre_id = 0
    with open(path) as annot_file:
        annotreader = csv.reader(annot_file, delimiter=',', quotechar='|')        
        row_count = 0
        for row in annotreader:
            if row==[]:
                continue
            if row_count==0:
                row_count += 1
                continue
            annotations[row[0]] = {'movie_title':row[2], 'genre':row[1].split('|')}
            valid_labels = []
            for genre in row[1].split('|'):
                G = genre.replace(' ', '')
                if (not G in genre_list.keys()) and (G in possible_genres):
                    genre_list[G] = genre_id
                    genre_id += 1
                if G in possible_genres:
                    valid_labels.append(genre)
            annotations[row[0]]['genre'] = valid_labels
            row_count += 1
            
    return annotations, genre_list
