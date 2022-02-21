#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 02:33:12 2022

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import lib.misc as misc
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
import os
import datetime
from sklearn.preprocessing import StandardScaler




def print_results(PARAMS, fold, suffix, **kwargs):
    if not suffix=='':
        opFile = PARAMS['opDir'] + '/Performance_' + suffix + '.csv'
    else:
        opFile = PARAMS['opDir'] + '/Performance.csv'

    linecount = 0
    if os.path.exists(opFile):
        with open(opFile, 'r', encoding='utf8') as fid:
            for line in fid:
                linecount += 1    
    
    fid = open(opFile, 'a+', encoding = 'utf-8')
    heading = 'fold'
    values = str(fold)
    for i in range(len(kwargs)):
        heading = heading + '\t' + np.squeeze(kwargs[str(i)]).tolist().split(':')[0]
        values = values + '\t' + np.squeeze(kwargs[str(i)]).tolist().split(':')[1]

    if linecount==0:
        fid.write(heading + '\n' + values + '\n')
    else:
        fid.write(values + '\n')
        
    fid.close()




if __name__ == '__main__':
    PARAMS = {
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            'dataset_name': 'Moviescope',
            'method1': 'LMS',
            'method1_path': './results/Moviescope/Best/Proposed_2DCNN_Attention/Attention_LMS/',
            'method2': 'CBoW_SMPred_Stats',
            'method2_path': './results/Moviescope/Best/Proposed_2DCNN_Attention/Attention_CBoW_SMPred_Stats/',
            'genre_list': {'Action':1, 'Fantasy':1, 'Sci-Fi':2, 'Thriller':3, 'Romance':4, 'Animation':5, 'Comedy':6, 'Family':7, 'Mystery':8, 'Drama':9, 'Crime': 10, 'Horror':11, 'Biography':12},
            
            # 'dataset_name': 'EmoGDB',
            # 'method1': 'CBoW_SMPred_Stats',
            # 'method1_path': './results/EmoGDB/Cross_Dataset_Performance/Attention_CBoW_SMPred_Stats_STL_30s_proposed/', # CBoW+SMPred+Stats
            # 'method2': 'LMS',
            # 'method2_path': './results/EmoGDB/Cross_Dataset_Performance/Attention_LMS_STL_30s_proposed/', # LMS
            # # 'genre_list': {'Action':0, 'Thriller':1, 'Romance':2, 'Comedy':3, 'Drama':4, 'Horror':5},
            # 'genre_list':{'Comedy': 0, 'Horror': 1, 'Romance': 2, 'Action': 3, 'Thriller': 4, 'Drama': 5},
            }
    PARAMS['opDir'] = './results/' + PARAMS['dataset_name'] + '/' + PARAMS['today'] + '/CrossDataset_Ensemble_' + PARAMS['method1'] + '_' + PARAMS['method2'] + '/'
    if not os.path.exists(PARAMS['opDir']):
        os.makedirs(PARAMS['opDir'])
    misc.print_configuration(PARAMS)

    Test_Params_Baseline = misc.load_obj(PARAMS['method1_path'], 'Test_Params_fold0')
    # Predictions_Baseline = Test_Params_Baseline['Predictions']
    # test_label = Test_Params_Baseline['test_label']
    Predictions_Baseline = Test_Params_Baseline['Trailer_Metrics']['Prediction']
    # Predictions_Baseline = StandardScaler().fit_transform(Predictions_Baseline)
    test_label = Test_Params_Baseline['Trailer_Metrics']['Groundtruth']
    
    Test_Params_Proposed = misc.load_obj(PARAMS['method2_path'], 'Test_Params_fold0')
    Predictions_Proposed = Test_Params_Proposed['Trailer_Metrics']['Prediction']
    # Predictions_Proposed = StandardScaler().fit_transform(Predictions_Proposed)
    # test_label = Test_Params_Proposed['Trailer_Metrics']['Groundtruth']

    best_perf = [0,0,0]
    best_alpha = 0
    Trailer_Metrics = {'P_curve':{}, 'R_curve':{}, 'AUC':{}, 'average_precision':{}, 'Precision':{}, 'Recall':{}, 'F1_score':{}}
    for alpha  in np.arange(0.0, 1.01, 0.01):
        alpha = np.round(alpha, 2)
        print('\nalpha=', alpha)
        Predictions = None
        Predictions = np.add(alpha*Predictions_Baseline, (1-alpha)*Predictions_Proposed)
        # Predictions = np.multiply(Predictions_Baseline**alpha, Predictions_Proposed**(1-alpha))
        
        # print('Trailer_pred: ', np.shape(Predictions), np.shape(test_label))
        macro_avg_auc = 0
        macro_weighted_avg_auc = 0
        for genre_i in PARAMS['genre_list'].keys():
            pred = Predictions[:, PARAMS['genre_list'][genre_i]]
            gt = test_label[:, PARAMS['genre_list'][genre_i]]
            precision_curve, recall_curve, threshold = precision_recall_curve(gt, pred)
            fscore_curve = np.divide(2*np.multiply(precision_curve, recall_curve), np.add(precision_curve, recall_curve)+1e-10)
            Trailer_Metrics['Precision'][genre_i] = np.round(np.mean(precision_curve),4)
            Trailer_Metrics['Recall'][genre_i] = np.round(np.mean(recall_curve),4)
            Trailer_Metrics['F1_score'][genre_i] = np.round(np.mean(fscore_curve),4)
            Trailer_Metrics['P_curve'][genre_i] = precision_curve
            Trailer_Metrics['R_curve'][genre_i] = recall_curve
            # Trailer_Metrics['AUC'][genre_i] = np.round(auc(recall_curve, precision_curve)*100,2)
            Trailer_Metrics['AUC'][genre_i] = np.round(average_precision_score(y_true=gt, y_score=pred, average='macro')*100,2)
            macro_avg_auc += Trailer_Metrics['AUC'][genre_i]
            # macro_weighted_avg_auc += genre_freq[genre_i]*Trailer_Metrics['AUC'][genre_i]
        
        micro_avg_precision_curve, micro_avg_recall_curve, threshold = precision_recall_curve(test_label.ravel(), Predictions.ravel())
        Trailer_Metrics['P_curve']['micro_avg'] = micro_avg_precision_curve
        Trailer_Metrics['R_curve']['micro_avg'] = micro_avg_recall_curve
        ap_macro = np.round(average_precision_score(y_true=test_label, y_score=Predictions, average='macro')*100,2)
        ap_micro = np.round(average_precision_score(y_true=test_label, y_score=Predictions, average='micro')*100,2)
        ap_samples = np.round(average_precision_score(y_true=test_label, y_score=Predictions, average='samples')*100,2)
        ap_weighted = np.round(average_precision_score(y_true=test_label, y_score=Predictions, average='weighted')*100,2)
        
        # curr_perf = ap_macro
        # curr_perf = ap_micro
        curr_perf = ap_samples
        # if (ap_macro>=best_perf[0]) and (ap_micro>=best_perf[1]) and (ap_samples>=best_perf[2]):
        # if ap_macro>=best_perf[0]:
        # if ap_micro>=best_perf[1]:
        if ap_samples>=best_perf[2]:
            print(f'Previous best: {best_perf}, New best: {curr_perf}, alpha={alpha}')
            Trailer_Metrics['best_alpha'] = alpha
            best_perf[0] = ap_macro
            best_perf[1] = ap_micro
            best_perf[2] = ap_samples
            Trailer_Metrics['AUC']['macro_avg'] = np.round(macro_avg_auc/len(PARAMS['genre_list']),4)
            Trailer_Metrics['AUC']['micro_avg'] = np.round(auc(micro_avg_recall_curve, micro_avg_precision_curve),4)
            Trailer_Metrics['AUC']['macro_avg_weighted'] = np.round(macro_weighted_avg_auc,4)
    
            Trailer_Metrics['average_precision']['macro'] = ap_macro
            Trailer_Metrics['average_precision']['micro'] = ap_micro
            Trailer_Metrics['average_precision']['samples'] = ap_samples
            Trailer_Metrics['average_precision']['weighted'] = ap_weighted        
    print('best alpha: ', Trailer_Metrics['best_alpha'])
    print('AUC (macro-avg): ', Trailer_Metrics['AUC']['macro_avg'])
    print('AUC (micro-avg): ', Trailer_Metrics['AUC']['micro_avg'])
    print('AUC (macro-avg weighted): ', Trailer_Metrics['AUC']['macro_avg_weighted'])        
    print('AP (macro): ', Trailer_Metrics['average_precision']['macro'])
    print('AP (micro): ', Trailer_Metrics['average_precision']['micro'])
    print('AP (samples): ', Trailer_Metrics['average_precision']['samples'])
    print('AP (weighted): ', Trailer_Metrics['average_precision']['weighted'])

    for genre_i in PARAMS['genre_list'].keys():
        # print(genre_i, 'AUC=', Trailer_Metrics['AUC'][genre_i])
        result = {
            '0':'best alpha:'+str(Trailer_Metrics['best_alpha']),
            '1':'genre:'+genre_i,
            '2':'AUC (macro):'+str(Trailer_Metrics['AUC']['macro_avg']),
            '3':'AUC (micro):'+str(Trailer_Metrics['AUC']['micro_avg']),
            '4':'AUC (weighted):'+str(Trailer_Metrics['AUC']['macro_avg_weighted']),
            '5':'AUC:'+str(Trailer_Metrics['AUC'][genre_i]),
            '6':'Precision:'+str(Trailer_Metrics['Precision'][genre_i]),
            '7':'Recall:'+str(Trailer_Metrics['Recall'][genre_i]),
            '8':'F1_score:'+str(Trailer_Metrics['F1_score'][genre_i]),
            '9':'AP (macro):'+str(Trailer_Metrics['average_precision']['macro']),
            '10':'AP (micro):'+str(Trailer_Metrics['average_precision']['micro']),
            '11':'AP (samples):'+str(Trailer_Metrics['average_precision']['samples']),
            '12':'AP (weighted):'+str(Trailer_Metrics['average_precision']['weighted']),
            }
        print_results(PARAMS, 0, '', **result)
