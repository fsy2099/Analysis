#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 13:47:05 2023

@author: shiyi
This script used to do paired samples T-test on the AMUA data to figure out hte difference between the reponse{0-200ms} and the baseline{300-500ms} 
if there is significant difference
the unit will be selected
"""

import numpy as np
import os 
#from matplotlib import pyplot as plt
from scipy import stats
from scipy.signal import butter, filtfilt
#%%
Sig_path = 'C:\\Users\\shiyi\\Documents\\1_Project\\ENVvsPT\\data\\'
#results_path = '/disks/CIdata/1_ENVvsPT/2022_08_24/Fig/AMUA_trace/'
selction_path = 'C:\\Users\\shiyi\\Documents\\1_Project\\ENVvsPT\\data\\'

file_names = os.listdir(Sig_path)
sig_names = [file_name for file_name in file_names if all([x in file_name for x in [ "_OriginSigArray_AMUA_Mean.npy"]])]
print(sig_names)                
#%%
'''
For every multi-unit, every stimulis combination,
use paired sample T-test to test the difference between response and baseline
'''
Fs_down = 2000
response_start = 0.005 # dispose first 10 samples
response_end = 0.055
baseline_start = 0.45
Wn = 2*200/Fs_down
bLow,aLow = butter(2, Wn, 'lowpass')

for sig_name in sig_names:
    print(sig_name)
    amua_array = np.load(Sig_path+sig_name, allow_pickle = True)
    results_array = np.zeros((32, 2))
    selection_array = np.zeros((32))
    ntrials = amua_array.shape[-1]
    for cc in range(32):
        response_array = np.zeros((2, 3, 3, 3, ntrials))   
        baseline_array = np.zeros((2, 3, 3, 3, ntrials))
        for ff in range(2):
            for dd in range(3):
                for ii in range(3):
                    for jj in range(3):
                        amua = amua_array[cc, ff, dd, ii, jj, :, :]
                        amua = filtfilt(bLow,aLow, amua, axis=0, padlen=100)
#                        amua_response = np.amax(amua[1:int(Fs_down*response_end), :], 0)
                        amua_response = np.mean(amua[int(Fs_down*response_start):int(Fs_down*response_end), :], 0)
                        amua_baseline = np.mean(amua[int(Fs_down*baseline_start):], 0) 
                        response_array[ff, dd, ii, jj, :] = amua_response
                        baseline_array[ff, dd, ii, jj, :] = amua_baseline
                        
        response = np.reshape(response_array, (1, 2*3*3*3*ntrials))[0]
        baseline = np.reshape(baseline_array, (1, 2*3*3*3*ntrials))[0]
        results = stats.wilcoxon(response, baseline)
        results_array[cc, :] = results
        if results[1] < 0.05:
            selection_array[cc] = 1 
    
    np.save(selction_path+sig_name[:-14]+'_AMUA_T_results.npy',results_array)
#    selection_array[21] = 0
#    if sig_name[:8] == '20220822':
#        selection_array[4] = 0
#        selection_array[10] = 0
#        selection_array[15] = 0
    np.save(selction_path+sig_name[:-14]+'_AMUA_Selection.npy',selection_array)


#for cc in range(32):
#    print(selection_array[cc, 0, 1, :, :])
#%%
'''
calculate how many multi unit is considered as active
'''
selction_path = '/disks/CIdata/1_ENVvsPT/selection/ND/_AMUA_Selection/'
file_names = os.listdir(selction_path)
sig_names = [file_name for file_name in file_names if all([x in file_name for x in [ "_AMUA_Selection.npy"]])]
print(sig_names)
x = len(sig_names)
unit_total = 32*x
print(unit_total)
active_total = 0
for sig_name in sig_names:
    selection_array = np.load(selction_path+sig_name)
    active_total = active_total+int(sum(selection_array))
print(active_total)
pro = active_total/unit_total
print(pro)
