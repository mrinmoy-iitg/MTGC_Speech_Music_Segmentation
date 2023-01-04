#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 10:38:47 2022

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import os
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import sys



def get_tdnn_embeddings(fName):
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb", 
        savedir="pretrained_models/spkrec-xvect-voxceleb",
        run_opts={"device":"cuda"},
        )
    signal, fs = torchaudio.load(fName)
    signal_shape = list(torch.Tensor.size(signal))
    # print(f'\tsignal torch={signal_shape}')
    tot_embed = np.empty([])
    sig_i = 0
    sig_j = 0
    while sig_j<signal_shape[1]:
        sig_i = sig_j
        sig_j = np.min([sig_i+fs, signal_shape[1]])
        if (sig_j-sig_i)<fs:
            sig_i = sig_j-fs
        signal_1s = signal[0,sig_i:sig_j]
        # print(f'\tsignal_1s torch={torch.Tensor.size(signal_1s)}')
        signal_1s = signal_1s[None, :]
        # print(f'\tsignal_1s torch={torch.Tensor.size(signal_1s)}')
        embeddings = classifier.encode_batch(signal_1s).cpu().detach().numpy()
        embeddings = np.array(np.squeeze(embeddings), ndmin=2)
        if np.size(tot_embed)<=1:
            tot_embed = embeddings
        else:
            tot_embed = np.append(tot_embed, embeddings, axis=0)
        # print(f'\ttot_embed shape={tot_embed.shape}')
        # sys.exit(0)
    # print('')
        # 
    return tot_embed



def __init__():
    PARAMS = {
        # EEE-GPU
        # 'data_path': '/home1/PhD/mrinmoy.bhattacharjee/data/musan/',
        # 'dataset_name': 'musan',
        # 'classes':{0:'music', 1:'speech'},

        'data_path': '/home1/PhD/mrinmoy.bhattacharjee/data/Moviescope/',
        'dataset_name': 'Moviescope',
        'classes':{0:'wav'},

        'folder': '/home1/PhD/mrinmoy.bhattacharjee/MTGC_SMO/',
        'save_flag': True,
        'feat_type': 'X-Vectors',
        }
    
    return PARAMS



if __name__ == '__main__':
    PARAMS = __init__()

    PARAMS['output_folder'] = PARAMS['folder'] + '/' + PARAMS['dataset_name'] + '/' + PARAMS['feat_type'] + '/'
    if not os.path.exists(PARAMS['output_folder']):
        os.makedirs(PARAMS['output_folder'])

    for clNum in [0]: # PARAMS['classes'].keys():
        classname = PARAMS['classes'][clNum]
        PARAMS['opDir'] = PARAMS['output_folder'] + '/' + classname + '/'
        if not os.path.exists(PARAMS['opDir']):
            os.makedirs(PARAMS['opDir'])

        files = next(os.walk(PARAMS['data_path'] + '/' + classname + '/'))[2]
        print(f'files: {len(files)}')
        for fl in files:
            opFile = PARAMS['opDir'] + '/' + fl.split('.')[0] + '.npy'
            if os.path.exists(opFile):
                print(f'{fl} xvector exists')
                continue
            fName = PARAMS['data_path'] + '/' + classname + '/' + fl
            xvector = get_tdnn_embeddings(fName)
            np.save(opFile, xvector)
            print(f'fName={fName} xvector={np.shape(xvector)} type={type(xvector)}')

    print('\n\n\nGenerated X-Vector features')
    
