#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:38:13 2023

@author: shiyi
"""
import pandas as pd
from sys import platform
import numpy as np
import numpy.matlib
import os
import matplotlib  
from matplotlib import pyplot as plt
import sys
from numpy.lib.stride_tricks import as_strided
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, csd, windows, welch, get_window
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM


Sig_path = 'C:\\Users\\shiyi\\Documents\\1_Project\\ENVvsPT\\data\\'
#file_names = os.listdir(Sig_path)
#sig_names = [file_name for file_name in file_names if all([x in file_name for x in [ 'P3', "_OriginSigArray_AMUA.npy"]])]
#sig_name = sig_names[0]
#amua_array = np.load(Sig_path+sig_name)
Results_path = 'C:\\Users\\shiyi\\Documents\\1_Project\\ENVvsPT\\data\\'
selection_path = 'C:\\Users\\shiyi\\Documents\\1_Project\\ENVvsPT\\data\\'
#%%
def BaselineCorrection(data):
    data = data[:, :, 2:, :]
    for pITD in range(3):
        for eITD in range(3):
            for ntrials in range(data.shape[-1]):
                data[pITD, eITD, :, ntrials] -= np.mean(np.mean(data[pITD, eITD, :, :], -1)[:6])
    return(data)

def lowPassFilter(data, freq, Fs_down = 2000):
    Wn = 2*freq/Fs_down
    bLow,aLow = butter(2, Wn, 'lowpass')
    output = np.zeros((3, 3, data.shape[2], data.shape[3]))
    for ii in range(3):
        for jj in range(3):
            sig = data[ii, jj, :, :]
            sig = filtfilt(bLow,aLow, sig, axis=0, padlen=100)
            output[ii, jj, :, :] = sig
    return output

def AvgAMUA(data):
    sig = np.mean(np.mean(np.mean(data, 0), 0), -1)
    return sig

def plotAMUA(data):
    fig = plt.figure()
    plt.plot(data)
    return fig
     
#%%
selection_names = os.listdir(selection_path)
file_names = os.listdir(Sig_path)
sig_names = [file_name for file_name in file_names if all([x in file_name for x in [ "_OriginSigArray_AMUA.npy"]])]
Fs_down = 2000
for sig_name in sig_names:
# sig_name = sig_names[0]
    print(sig_name)
    selection_name = sig_name[:-4]+'_Selection.npy'
    amua_array = np.load(Sig_path+sig_name)
    selection_array = np.load(selection_path+selection_name)
    win_array = np.zeros((32, 2, 3, 2))
    for cc in range(32):
        if selection_array[cc] == 0:
            continue
        print('Ch'+ str(cc))
        for ff in range(2):
            for dd in range(3):
                print('PR'+str(ff)+'Dur'+str(dd))
                if selection_array[cc] == 0:
                    continue
                # baseline correction
                amua = BaselineCorrection(amua_array[cc, ff, dd])
                # 200Hz low pass filter
                amua = lowPassFilter(amua, 200)
                # response threshold = Avg(baseline)+3*sd(baseline)
                amua = amua[:, :, :-2, :]
                baseline = AvgAMUA(amua)[-100:]
                amua = AvgAMUA(amua)
    #            amua = np.concatenate(([0], amua))
                baseline_avg = np.mean(baseline)
                baseline_sd = np.std(baseline)
                thresh = baseline_avg+3*baseline_sd
                x = np.arange(0, 900)
#                if dd == 2:
#                plotAMUA(amua)
#                plt.plot(x, np.repeat(thresh, 900))
                                
                idx_start = 0
                idx_end = 0
                idxs = np.where(amua>=thresh)[0]
                if len(idxs) <= 10:
                    selection_array[cc] = 0
                    continue
                num = 5
                
                # find the time window the AMUA response larger than thresh 5 data points in row                
                for x1 in range(idxs.shape[0]):
                    y = 0
                    for a in range(num):
                        if idxs[x1]+a in idxs:
                            y=y+1
                    if y == num:
                        idx_start1 = idxs[x1]
                        break
                idxs_short = idxs[x1:]
                for x2 in range(idxs.shape[0]):
                    y = 0
                    for a in range(1, num+1, 1):
                        if idxs_short[x2]+a not in idxs_short:
                            y=y+1
                    if y == num:
                        idx_end1 = idxs_short[x2]
                        break
                
                # find the time window searching from peak
                peak_idx_inAmua = np.argmax(amua[:500])
                peak_idx_inIdxs = np.where(idxs==peak_idx_inAmua)[0][0]
                idxs_flip_front = np.flip(idxs[:peak_idx_inIdxs])
                idxs_back = idxs[peak_idx_inIdxs:]
                for y1 in range(len(idxs_flip_front)):
                    y = 0
                    for a in range(1, num+1, 1):
                        if idxs_flip_front[y1]-a not in idxs_flip_front:
                            y = y+1
                    if y == num:
                        idx_start2 = idxs_flip_front[y1]
#                        print(idx_start2)
                        break
                for y2 in range(len(idxs_back)):
                    y = 0
                    for a in range(1, num+1, 1):
                        if idxs_back[y2]+a not in idxs_back:
                            y = y+1
                    if y == num:
                        idx_end2 = idxs_back[y2]
#                        print(idx_end2)
                        break
                
                # compare two time window
                idx_start = min(idx_start1, idx_start2)
                idx_end = max(idx_end1, idx_end2)                
                print(idx_start)
                print(idx_end)
              
                
    #            [start, end] = FindWin(amua, amua_thresh)
                win_array[cc, ff, dd, 0] = idx_start
                win_array[cc, ff, dd, 1] = idx_end
    #            print([start, end])
    np.save(Results_path+sig_name[:-4]+'_Avg_Window.npy', win_array)
    np.save(selection_path+sig_name[:-9]+'_VE_Selection.npy',selection_array)
#%%
#'''
#check the distribution of IdxStart and IdxEnd
#'''
#AvgWin_names = os.listdir(Results_path)
#Idx_array = np.zeros((len(AvgWin_names), 32, 2))
#ff = 0
#dd = 0
#for nn in range(len(AvgWin_names)):
#    AvgWin_name = AvgWin_names[nn]
#    AvgWin_array = np.load(Results_path+AvgWin_name)
#    Idx_array[nn, :, :] = AvgWin_array[:, ff, dd, :]














