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
