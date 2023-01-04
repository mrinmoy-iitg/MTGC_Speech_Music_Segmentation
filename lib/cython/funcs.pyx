#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 17:52:56 2021

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati

"""

import numpy as np
cimport numpy as np


np.import_array()  # needed to initialize numpy-API

def compute_segment_predictions(pred, genre_list, smo_labels, Train_Params, fv):
    cdef np.ndarray[double, ndim=2] Genre_Pred = np.zeros((np.shape(pred)[0], len(genre_list)))
    
    for seg_i in range(np.shape(pred)[0]):
        g_pred_i = Train_Params[smo_labels[seg_i]]['model'].predict(np.array(fv[seg_i,:], ndmin=2))
        Genre_Pred[seg_i, :] = g_pred_i
    
    return Genre_Pred        


def extract_patches(FV, shape, patch_size, patch_shift):
    cdef int frmStart, frmEnd, i
    cdef int nFrames = shape[1]
    cdef int half_win = int(patch_size/2)
    cdef int numPatches = len(list(range(half_win, nFrames-half_win, patch_shift)))
    cdef np.ndarray[double, ndim=4] patches = np.zeros((numPatches, shape[0], patch_size, shape[2]))
    cdef int nPatch = 0
    cdef int numFrame = 0
    for i in range(half_win, nFrames-half_win, patch_shift):
        frmStart = i-half_win
        # frmEnd = i+half_win
        frmEnd = np.min([frmStart + patch_size, nFrames])
        if (frmEnd-frmStart)<patch_size:
            frmStart = frmEnd-patch_size
        patches[nPatch,:,:,:] = FV[:,frmStart:frmEnd,:].copy()
        nPatch += 1
        numFrame += patch_shift
    return patches



def speech_music_others_segmentation(high_confidence_thresh, sm_pred, nFrames):
    cdef np.ndarray[long, ndim=1] segmentation_labels = (np.ones(np.shape(sm_pred)[0])*2).astype(int)
    cdef np.ndarray[long, ndim=1] label_mapping = (np.cumsum([1]*len(segmentation_labels))*nFrames/len(segmentation_labels)).astype(int)
    cdef np.ndarray[long, ndim=1] frame_labels = np.zeros(nFrames).astype(int)
    cdef int frmStart, i

    segmentation_labels[np.squeeze(np.where(sm_pred[:,1]>=high_confidence_thresh))] = 1
    segmentation_labels[np.squeeze(np.where(sm_pred[:,1]<=(1-high_confidence_thresh)))] = 0
    segmentation_labels = segmentation_labels.astype(int)
    # print('segmentation_labels: ', np.shape(segmentation_labels))
    
    label_mapping[-1] = nFrames
    # print('label_mapping: ', np.shape(label_mapping))
    
    frmStart = 0
    for i in range(len(label_mapping)):
        frame_labels[frmStart:label_mapping[i]] = segmentation_labels[i]
        frmStart = label_mapping[i]
    # print('frame_labels: ', np.shape(frame_labels))
    
    return frame_labels
