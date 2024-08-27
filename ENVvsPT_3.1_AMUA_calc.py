#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 18:04:16 2022

@author: shiyi

Calculate AMUA
"""

from sys import platform
import numpy as np
import numpy.matlib
import os 
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import as_strided
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, csd, windows, welch, get_window, resample, lfilter
from matplotlib.gridspec import GridSpec

if platform == "darwin":  
    os.chdir('/Users/fei/Documents/CI_projects/20220127_Analysis')
    fname = '20220127_ENVvsPT_5_P10_1'   
elif platform == "linux":
    Sig_path = '/disks/CIdata/1_ENVvsPT/2_ND/2023_03_03/Results_Data/'
    results_path = '/disks/CIdata/1_ENVvsPT/2_ND/2023_03_03/Results_Data/'
elif platform == "win32":
    Sig_path = 'C:\\Users\\shiyi\\Documents\\1_Project\\ENVvsPT\\data\\'
    results_path = 'C:\\Users\\shiyi\\Documents\\1_Project\\ENVvsPT\\data\\'   
## Stimulus waveform    
#if platform == "darwin":
#    stimPath = '/Users/fei/Documents/CI_projects/StimData_v1/'
#elif platform == "linux":
#    stimPath = '/home/colliculus/ephys/4/CIproject/0_ENVvsPTephys/Analysis/'
#StimulusData = np.load(stimPath+'Stim_ENVvsFS_template.npy')
#%%
def AMUAFilterCoeffs(fs,lowpass=6000):
        nyq = 0.5*fs
        bBand,aBand = butter(2,(300/nyq, 6000/nyq),'bandpass')
        bLow,aLow = butter(2,(lowpass/nyq),'lowpass')
        bNotch, aNotch = iirnotch(50, 30, fs)
        return [[bBand, aBand], [bLow, aLow], [bNotch, aNotch]]
    
def calcAMUA(fs, ori_signal, Fs_downsample, padLen=300):
        coefs=AMUAFilterCoeffs(fs)
        bpCoefs=coefs[0]
        lpCoefs=coefs[1]
        NotchCoefs = coefs[2]
        insig = filtfilt(NotchCoefs[0], NotchCoefs[1], ori_signal, axis=0, padlen=padLen)
        insig = np.flip(insig)
        insig=filtfilt(bpCoefs[0],bpCoefs[1], insig, axis=0, padlen=padLen)
        insig=np.abs(insig)
        insig=filtfilt(lpCoefs[0],lpCoefs[1],insig,axis=0, padlen=padLen)
        insig = np.flip(insig)          
        # Fs_downsample
        # signal = resample_poly(insig, Fs_downsample, int(fs), axis=0)
        downsample_length = int((insig.shape[0]/fs)*Fs_downsample)
        signal=resample(insig,downsample_length)
        
        return signal

#%% 
Fs = 24414.0625
Fs_down = 2000
#nsamples_down = int(Fs_down*0.5)-1 
stiDur = [0.01, 0.05, 0.2]
stiRate = [900, 4500]
stiITD = [-0.1, 0, 0.1]
stienvITD = [-0.1, 0, 0.1]
file_names = os.listdir(Sig_path)
clean_names = [file_name for file_name in file_names if all([x in file_name for x in ["_CleanSig.npy"]])]
print(clean_names)
#%%
'''
calculate AMUA
'''
for x in range(len(clean_names)):
#x = 0
    fname = clean_names[x][:-13]
    print(fname)
    clean_array = np.load(results_path+clean_names[x])
    nsamples = clean_array.shape[-2]/Fs
    nsamples_down = int(Fs_down*nsamples)-1 
    amua_array = np.zeros((32, 2, 3, 3, 3, nsamples_down, clean_array.shape[-1]))
    for cc in range(32):
        for dd in range(3):
            for ii in range(3):
                for jj in range(3):
                    StimParam = [cc, stiDur[dd], stiITD[ii], stienvITD[jj]]
                    print(StimParam)
                    # calculate AMUA
                    for ff in range(2):
                        clean = clean_array[cc, ff, dd, ii, jj, 5:, :]
                        amua = calcAMUA(Fs, clean, Fs_down)
                        amua_array[cc, ff, dd, ii, jj, :, :] = amua[:nsamples_down, :]
    np.save(results_path+fname+'_AMUA.npy',amua_array)
    print('data saved')























    