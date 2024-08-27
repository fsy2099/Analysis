#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 15:16:31 2023

@author: shiyi
"""

from sys import platform
import numpy as np
import numpy.matlib
import os 
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import as_strided
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, csd, windows, welch, get_window
# give signal path
if platform == "darwin":  
    Sig_path = 'C:\\Users\\shiyi\\Documents\\ENVvsPT data\\ND\\2022_01_25\\'
    results_path = 'C:\\Users\\shiyi\\Documents\\ENVvsPT data\\ND\\2022_01_25\\'   
elif platform == "linux":
    Sig_path = '/disks/CIdata/1_ENVvsPT/1_NH/2022_08_22/Results_Data/'
#    Sig_path = '/media/shiyi/CIdata/2023_02_16/'
    results_path = '/disks/CIdata/1_ENVvsPT/1_NH/2022_08_22/Results_Data/'
elif platform == "win32":
    Sig_path = 'C:\\Users\\shiyi\\Documents\\1_Project\\ENVvsPT\\example\\'
    results_path = 'C:\\Users\\shiyi\\Documents\\1_Project\\ENVvsPT\\example\\'   
    
#    results_path = '/media/shiyi/CIdata/2023_02_16/'
# load stimulus waveform    
# if platform == "darwin":z
#     stimPath = '/Users/fei/Documents/CI_projects/StimData_v1/'
# elif platform == "linux":
#     stimPath = '/disks/CIdata/1_ENVvsPT/StimuliData/'
# #    stimPath = '/media/shiyi/CIdata/StimuliData/'
# StimulusData = np.load(stimPath+'Stim_ENVvsFS_template.npy')
#%%
file_names = os.listdir(Sig_path)
sig_names = [file_name for file_name in file_names if all([x in file_name for x in ["_OriginSigArray.npy"]])]
print(sig_names)
Fs = 24414.0625
stiDur = [0.01, 0.05, 0.2]
stiRate = [900, 4500]
stiITD = [-0.1, 0, 0.1]
stienvITD = [-0.1, 0, 0.1]
#idx_05 = round(24414.0625*0.05)
#WienerFilterOrder = 25
for sig_name in sig_names: 
    # sig_name = sig_names[0]
    sig_array = np.load(Sig_path+sig_name)
    print(sig_name)
    nsamples = sig_array.shape[-2]
    nchannel = sig_array.shape[0]
    ntrials = sig_array.shape[-1]
    clean_array = np.zeros((32,len(stiRate),len(stiDur),len(stiITD),len(stienvITD),nsamples,ntrials),dtype = 'float32')
    for cc in range(nchannel):
        for ff in range(2):
            for dd in range(3):
                for ii in range(3):
                    for jj in range(3):
                        sig_temp = sig_array[cc, ff, dd, ii, jj, :, :] 
                        sig_avg = np.mean(sig_temp, -1)
                        sig_avg_array = np.array([sig_avg] * ntrials).T
                        sig_clean = sig_temp-sig_avg_array
                        clean_array[cc, ff, dd, ii, jj, :, :] = sig_clean
    np.save(results_path+sig_name[:-4]+'_CleanSig_Mean.npy',clean_array)
    


                        
                        
                        
                        
                        
