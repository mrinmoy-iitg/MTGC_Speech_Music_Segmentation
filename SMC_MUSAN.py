#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 12:51:23 2018
Updated on Tue Nov 16 17:27:26 2021

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import os
import datetime
import sys
import time
import lib.misc as misc
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
# from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.initializers import RandomNormal, Constant
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.utils import to_categorical



# define base model
# def dnn_model(input_dim, output_dim):
#     # create model
    
#     input_layer = Input((input_dim,))
    
#     layer1_size = input_dim
#     x = Dense(layer1_size, input_dim=(input_dim,), kernel_initializer=he_uniform(0))(input_layer)
#     x = BatchNormalization(axis=-1)(x)
#     x = Activation('relu')(x)
#     x = Dropout(0.4)(x)
    
#     layer2_size = layer1_size*2
#     x = Dense(layer2_size, kernel_initializer=he_uniform(0))(x)
#     x = BatchNormalization(axis=-1)(x)
#     x = Activation('relu')(x)
#     x = Dropout(0.4)(x)

#     layer3_size = int(layer2_size*2/3)
#     x = Dense(layer3_size, kernel_initializer=he_uniform(0))(x)
#     x = BatchNormalization(axis=-1)(x)
#     x = Activation('relu')(x)
#     x = Dropout(0.4)(x)

#     layer4_size = int(layer3_size/2)
#     x = Dense(layer4_size, kernel_initializer=he_uniform(0))(x)
#     x = BatchNormalization(axis=-1)(x)
#     x = Activation('relu')(x)
#     x = Dropout(0.4)(x)

#     layer5_size = int(layer4_size/3)
#     x = Dense(layer5_size, kernel_initializer=he_uniform(0))(x)
#     x = BatchNormalization(axis=-1)(x)
#     x = Activation('relu')(x)
#     x = Dropout(0.4)(x)
    
#     output_layer = Dense(output_dim, activation='softmax')(x)

#     model = Model(input_layer, output_layer)
    
#     learning_rate = 0.0001
#     adam = Adam(lr=learning_rate)

#     optimizerName = 'Adam'
#     # Compile model
#     model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

#     return model, optimizerName, learning_rate




def dnn_model(input_dim, output_dim):
    # create model
    
    input_layer = Input((input_dim,))
    
    layer1_size = input_dim
    dense_1 = Dense(layer1_size, input_dim=(input_dim,), kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(input_layer)
    batchnorm_1 = BatchNormalization()(dense_1)
    dropout_1 = Dropout(0.4)(batchnorm_1)
    activation_1 = Activation('relu')(dropout_1)
    
    layer2_size = layer1_size*2
    dense_2 = Dense(layer2_size, kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(activation_1)
    batchnorm_2 = BatchNormalization()(dense_2)
    dropout_2 = Dropout(0.4)(batchnorm_2)
    activation_2 = Activation('relu')(dropout_2)

    layer3_size = int(layer2_size*2/3)
    dense_3 = Dense(layer3_size, kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(activation_2)
    batchnorm_3 = BatchNormalization()(dense_3)
    dropout_3 = Dropout(0.4)(batchnorm_3)
    activation_3 = Activation('relu')(dropout_3)

    layer4_size = int(layer3_size/2)
    dense_4 = Dense(layer4_size, kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(activation_3)
    batchnorm_4 = BatchNormalization()(dense_4)
    dropout_4 = Dropout(0.4)(batchnorm_4)
    activation_4 = Activation('relu')(dropout_4)

    layer5_size = int(layer4_size/3)
    dense_5 = Dense(layer5_size, kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(activation_4)
    batchnorm_5 = BatchNormalization()(dense_5)
    dropout_5 = Dropout(0.4)(batchnorm_5)
    activation_5 = Activation('relu')(dropout_5)
    
    output_layer = Dense(output_dim, kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(activation_5)
    output_layer = Activation('softmax')(output_layer)

    model = Model(input_layer, output_layer)
    
    learning_rate = 0.0001
    adam = Adam(lr=learning_rate)

    optimizerName = 'Adam'
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model, optimizerName, learning_rate





def train_model(PARAMS, data_dict, model, weightFile, logFile):
    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, restore_best_weights=True, min_delta=0.01, patience=5)
    mcp = ModelCheckpoint(weightFile, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')
    csv_logger = CSVLogger(logFile)

    trainingTimeTaken = 0
    start = time.process_time()
    train_data = data_dict['train_data']
    val_data = data_dict['val_data']
    OHE_trainLabel = to_categorical(data_dict['train_label'], num_classes=2)
    OHE_valLabel = to_categorical(data_dict['val_label'], num_classes=2)
    print('train data: ', np.shape(train_data), np.shape(OHE_trainLabel), np.sum(OHE_trainLabel, axis=0))
    print('val data: ', np.shape(val_data), np.shape(OHE_valLabel), np.sum(OHE_valLabel, axis=0))
    
    # Train the model
    History = model.fit(
            x=train_data,
            y=OHE_trainLabel, 
            epochs=PARAMS['epochs'],
            batch_size=PARAMS['batch_size'], 
            verbose=1,
            validation_data = (val_data, OHE_valLabel),
            callbacks=[csv_logger, es, mcp],
            shuffle=True,
            )

    trainingTimeTaken = time.process_time() - start
    print('Time taken for model training: ',trainingTimeTaken)

    return model, trainingTimeTaken, History





'''
This function is the driver function for learning and evaluating a DNN model. 
The arguments are passed as a dictionary object initialized with the required
data. It returns a dictinary that contains the trained model and other required
information.
'''
def perform_training(PARAMS, data_dict): # Updated on 25-05-2019
    modelName = '@'.join(PARAMS['modelName'].split('.')[:-1]) + '.' + PARAMS['modelName'].split('.')[-1]
    weightFile = modelName.split('.')[0] + '.h5'
    architechtureFile = modelName.split('.')[0] + '.json'
    paramFile = modelName.split('.')[0] + '_params.npz'
    logFile = modelName.split('.')[0] + '_log.csv'

    modelName = '.'.join(modelName.split('@'))
    weightFile = '.'.join(weightFile.split('@'))
    architechtureFile = '.'.join(architechtureFile.split('@'))
    paramFile = '.'.join(paramFile.split('@'))
    logFile = '.'.join(logFile.split('@'))
    
    output_dim = len(PARAMS['classes'])
    print(output_dim)

    print('Weight file: ', weightFile, PARAMS['input_dim'], output_dim)
    if not os.path.exists(paramFile):
        model, optimizerName, learning_rate = dnn_model(PARAMS['input_dim'], output_dim)
        print(model.summary())
        
        model, trainingTimeTaken, History = train_model(PARAMS, data_dict, model, weightFile, logFile)
        if PARAMS['save_flag']:
            # Save the weights
            model.save_weights(weightFile)
            # Save the model architecture
            with open(architechtureFile, 'w') as f:
                f.write(model.to_json())
            np.savez(paramFile, lr=str(learning_rate), TTT=str(trainingTimeTaken))
    else:
        learning_rate = float(np.load(paramFile)['lr'])
        trainingTimeTaken = float(np.load(paramFile)['TTT'])
        optimizerName = 'Adam'

        # Model reconstruction from JSON file
        with open(architechtureFile, 'r') as f:
            model = model_from_json(f.read())
        # Load weights into the new model
        model.load_weights(weightFile)
        opt = optimizers.Adam(lr=learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        print('DNN model exists! Loaded. Training time required=',trainingTimeTaken)
        # print(model.summary())
    
    Train_Params = {
            'model': model,
            'trainingTimeTaken': trainingTimeTaken,
            'learning_rate': learning_rate,
            'optimizerName': optimizerName,
            'paramFile': paramFile,
            'architechtureFile': architechtureFile,
            'weightFile': weightFile,
            }
    
    return Train_Params




def test_model(PARAMS, data_dict, Train_Params):
    loss = 0
    performance = 0
    testingTimeTaken = 0
    PtdLabels = []
    test_data = data_dict['test_data']
    
    start = time.process_time()
    OHE_testLabel = to_categorical(data_dict['test_label'], num_classes=2)
    loss, performance = Train_Params['model'].evaluate(x=test_data, y=OHE_testLabel)
    Predictions = Train_Params['model'].predict(test_data)
    PtdLabels = np.argmax(Predictions, axis=1)
    GroundTruth = data_dict['test_label']
        
    testingTimeTaken = time.process_time() - start
    print('Time taken for model testing: ',testingTimeTaken)
    ConfMat, precision, recall, fscore = misc.getPerformance(PtdLabels, GroundTruth, labels=[0,1])
    
    Test_Params = {
        'loss':loss, 
        'performance': performance, 
        'testingTimeTaken': testingTimeTaken, 
        'ConfMat': ConfMat, 
        'precision': precision,
        'recall': recall,
        'fscore': fscore, 
        'PtdLabels': PtdLabels, 
        'Predictions': Predictions,
        }    
    
    return Test_Params



def start_GPU_session():
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.4)
    config = tf.compat.v1.ConfigProto(
        device_count={'GPU': 1 , 'CPU': 1}, 
        gpu_options=gpu_options,
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        )
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)



def reset_TF_session():
    tf.compat.v1.keras.backend.clear_session()



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




def __init__():
    PARAMS = {
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            
            # 'path': './features/MTGC_SMO/musan/1000ms_10ms_5ms_10PT/', # EEE-GPU
            # 'featName': 'MSD-ASPT-LSPT',
            # 'input_dim': 40,
            # 'featName': 'CBoW-ASPT-LSPT_mix5',
            # 'input_dim': 200, # MSD-ASPT=MSD-LSPT=20, CBoW-ASPT=CBoW-LSPT=100, MSD-ASPT-LSPT=40, CBoW-ASPT-LSPT=200
            'path': '/home1/PhD/mrinmoy.bhattacharjee/MTGC_SMO/musan/',
            'featName': 'X-Vectors',
            'input_dim': 512,

            'CV_folds': 3,
            'fold': 0,
            'save_flag': True,
            'data_generator': False,
            'use_GPU': True,
            'GPU_session':None,
            'classes':{0:'music', 1:'speech'},
            'epochs': 100,
            'batch_size': 64,
            'W': 1000, # interval size in ms
            'W_shift': 1000, # interval shift in ms
            }

    PARAMS['dataset_name'] = 'musan' # list(filter(None,PARAMS['folder'].split('/')))[-1]
    PARAMS['CV_path_train'] = './cross_validation_info/' + PARAMS['dataset_name'] + '/'    
    if not os.path.exists(PARAMS['CV_path_train']+'/cv_file_list.pkl'):
        print('Fold division of files for cross-validation not done!')
        sys.exit(0)
    PARAMS['cv_file_list'] = misc.load_obj(PARAMS['CV_path_train'], 'cv_file_list')
    
    n_classes = len(PARAMS['classes'])
    DT_SZ = 0
    for clNum in PARAMS['classes'].keys():
        classname = PARAMS['classes'][clNum]
        DT_SZ += PARAMS['cv_file_list']['total_duration'][classname] # in Hours
    DT_SZ *= 3600*1000 # in msec
    tr_frac = ((PARAMS['CV_folds']-1)/PARAMS['CV_folds'])*0.7
    vl_frac = ((PARAMS['CV_folds']-1)/PARAMS['CV_folds'])*0.3
    ts_frac = (1/PARAMS['CV_folds'])
    PARAMS['TR_STEPS'] = int(np.floor(DT_SZ/PARAMS['W_shift'])*tr_frac/(n_classes*PARAMS['batch_size']))
    PARAMS['V_STEPS'] = int(np.floor(DT_SZ/PARAMS['W_shift'])*vl_frac/(n_classes*PARAMS['batch_size']))
    PARAMS['TS_STEPS'] = int(np.floor(DT_SZ/PARAMS['W_shift'])*ts_frac/(n_classes*PARAMS['batch_size']))
    print('TR_STEPS: %d, \tV_STEPS: %d,  \tTS_STEPS: %d\n'%(PARAMS['TR_STEPS'], PARAMS['V_STEPS'], PARAMS['TS_STEPS']))

    return PARAMS



if __name__ == '__main__':
    PARAMS = __init__()

    for PARAMS['fold'] in range(PARAMS['CV_folds']):
        PARAMS['train_files'], PARAMS['test_files'] = get_train_test_files(PARAMS['cv_file_list'], PARAMS['CV_folds'], PARAMS['fold'])

        ''' Only for CBoW features '''
        if PARAMS['featName'].startswith('CBoW'):
            PARAMS['folder'] = PARAMS['path'] + '/' + PARAMS['featName'] + '/fold' + str(PARAMS['fold']) + '/'
        else:
            PARAMS['folder'] = PARAMS['path'] + '/' + PARAMS['featName'] + '/'
        print('folder: ', PARAMS['folder'])
        ''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''

        ''' Initializations '''
        PARAMS['output_folder'] = './results/' + PARAMS['dataset_name'] + '/' + PARAMS['today'] + '/'
        if not os.path.exists(PARAMS['output_folder']):
            os.makedirs(PARAMS['output_folder'])

        PARAMS['opDir'] = PARAMS['output_folder'] + '/' + PARAMS['featName'] + '/'
        if not os.path.exists(PARAMS['opDir']):
            os.makedirs(PARAMS['opDir'])
        misc.print_configuration(PARAMS)


        if PARAMS['use_GPU']:
            PARAMS['GPU_session'] = start_GPU_session()

        ''' Load data '''
        if not PARAMS['data_generator']:
            data_dict = misc.get_data(PARAMS)
        else:
            data_dict = {'train_data':np.empty, 'train_label':np.empty, 'val_data':np.empty, 'val_label':np.empty, 'test_data':np.empty, 'test_label':np.empty}
        print('Labels: ', np.unique(data_dict['train_label'], return_counts=True), np.unique(data_dict['val_label'], return_counts=True), np.unique(data_dict['test_label'], return_counts=True))
    
        ''' Set training parameters '''
        print('Input dim: ', PARAMS['input_dim'])
        PARAMS['modelName'] = PARAMS['opDir'] + '/fold' + str(PARAMS['fold']) + '_model.xyz'

        Train_Params = perform_training(PARAMS, data_dict)
        if not os.path.exists(PARAMS['opDir']+'/Test_Params_fold'+str(PARAMS['fold'])+'.pkl'):
            Test_Params = test_model(PARAMS, data_dict, Train_Params)
            misc.save_obj(Test_Params, PARAMS['opDir'], 'Test_Params_fold'+str(PARAMS['fold']))
        else:
            Test_Params = misc.load_obj(PARAMS['opDir'], 'Test_Params_fold'+str(PARAMS['fold']))

        print('Test accuracy=', Test_Params['fscore'])

        kwargs = {
                '0':'training_time:'+str(Train_Params['trainingTimeTaken']),
                '1':'loss:'+str(Test_Params['loss']),
                '2':'performance:'+str(Test_Params['performance']),
                '3':'precision_mu:'+str(Test_Params['precision'][0]),
                '4':'recall_mu:'+str(Test_Params['recall'][0]),
                '5':'F1_mu:'+str(Test_Params['fscore'][0]),
                '6':'precision_sp:'+str(Test_Params['precision'][1]),
                '7':'recall_sp:'+str(Test_Params['recall'][1]),
                '8':'F1_sp:'+str(Test_Params['fscore'][1]),
                '9':'F1_avg:'+str(Test_Params['fscore'][2]),
                }
        misc.print_results(PARAMS, '', **kwargs)
        Train_Params = None
        Test_Params = None

        if PARAMS['use_GPU']:
            reset_TF_session()

