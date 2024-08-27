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

#def plotRejectMark(X):
#    x = plt.figure(figsize=(10,15))
#    plt.xticks(np.arange(1, 33, 1))
#    plt.title('reject mark')
#    #plt.subplot(1,2,1)
#    y = 0
#    for dd in range(3):
#        for ii in range(3):
#            for jj in range(3):
#                y = y+1
#                for cc in range(32):
#                    if reject_mark[cc, dd, ii, jj] == 0:                        
#                        plt.plot(cc+1,y, 'ko')
#                    elif reject_mark[cc, dd, ii, jj] == 1:
#                        plt.plot(cc+1,y, 'ro')
#                    elif reject_mark[cc, dd, ii, jj] == 2:
#                        plt.plot(cc+1,y, 'bo')
#                    else:
#                        plt.plot(cc+1,y, 'go')
#    return x
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
cosidering some clean signal padding 1-3 zeros at begining to align up
in case the padding will affect the AMUA and frequency domain results
fllowing is checking code
'''
## example data from '20220127_ENVvsPT_5_P3_1_OriginSigArray_CleanSig.npy'
#xx = 0
#clean_name = clean_names[xx]
#clean_sig = np.load(Sig_path+clean_name)
## example stimulus parameter[0ch, 900pps, 0.01s, -0.1s, 0.1s]
#EndIdx = int(Fs*0.01)
#clean = clean_sig[0, 0, 0, 0, 2, :, :]
#idx = np.where(clean[0, :] == 0)
#print(idx)
## example trial number [2, 4, 8]
#plt.figure()
#plt.plot(clean[:EndIdx+50, 8])
## AMUA of clean sig
#amua_long = calcAMUA(Fs, clean[1:, 2], Fs_down)
#amua_short =  calcAMUA(Fs, clean[2:, 2], Fs_down)
#plt.figure()
#plt.plot(amua_long)
#plt.plot(amua_short)
#%%
'''
check how filtfilt function affect the amua results
'''
#clean_name = clean_names[1]
#print(clean_name)
#clean_array = np.load(Sig_path+clean_name)
#coefs = AMUAFilterCoeffs(Fs)
#padLen = 300
#clean_temp = clean_array[10, 0, 1, 0, 1, 5:, 5]
## see wether the filtfilt affect the resultes
#insig1 = filtfilt(coefs[0][0], coefs[0][1], clean_temp, axis=0, padlen=padLen)
#insig2 = lfilter(coefs[0][0], coefs[0][1], clean_temp, axis=0)
#
#fig = plt.figure(figsize=(20, 5))
#gs = GridSpec(nrows = 1, ncols = 3)
#ax = fig.add_subplot(gs[0, 0])
#ax.set_title('origin clean signal ')
#ax.plot(clean_temp)
#ax = fig.add_subplot(gs[0, 1])
#ax.set_title('lfilter')
#ax.plot(insig2)
#ax.set_ylim(-0.00005, 0.000055)  
#ax = fig.add_subplot(gs[0, 2])
#ax.set_title('filtfilt')
#ax.plot(insig1)  
#ax.set_ylim(-0.00005, 0.000055)  
#
## see wether the north filter affect the results
#insig1 = filtfilt(coefs[2][0], coefs[2][1], clean_temp, axis=0, padlen=padLen)
#insig1 = filtfilt(coefs[0][0], coefs[0][1], insig1, axis=0, padlen=padLen)
#insig1 = filtfilt(coefs[1][0], coefs[1][1], insig1, axis=0, padlen=padLen)
#insig2 = filtfilt(coefs[0][0], coefs[0][1], clean_temp, axis=0, padlen=padLen)
#insig2 = filtfilt(coefs[1][0], coefs[1][1], insig2, axis=0, padlen=padLen)
#
#fig = plt.figure(figsize=(10, 5))
#gs = GridSpec(nrows = 1, ncols = 2)
#ax = fig.add_subplot(gs[0, 0])
#ax.set_title('Notch filter ')
#ax.plot(insig1[:550])
#ax = fig.add_subplot(gs[0, 1])
#ax.set_title('Without Notch filter ')
#ax.plot(insig2[:550])
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
#%%
'''
check old AMUA data
see if there is drop down peak at begining 
'''
#results_path = '/disks/CIdata/2022_03_02/Results_Data/'
#file_names = os.listdir(results_path)
#amua_names = [file_name for file_name in file_names if all([x in file_name for x in ["_AMUA.npy"]])]
#print(amua_names)
#fname = amua_names[1]
#amua_array = np.load(results_path+fname)
#cc = 0
##%%
## example stimulus combination [900pps, 0.01s, -0.1, 0.1]
#print(cc)
#amua = amua_array[cc, 0, 0, 0, 1, :1000, :]
#plt.figure()
#plt.plot(amua)
#cc = cc+1
#%%
'''
plot:
    1. low pass(<200Hz) filtered AMUA trace, 3ENV*3PT    
'''
#Fs_down = 2000
#Wn = 2*200/Fs_down
#bLow,aLow = butter(2, Wn, 'lowpass')
#amua_uv = amua_array*1000000
#t = np.arange(0, 0.1, 1/Fs_down)
#ff = 0
#dd = 0
#for cc in range(32):
#    fig = plt.figure(figsize=(10,15), frameon = False)
#    gs = fig.add_gridspec(3, 3, hspace=0.25, wspace=0.15)
#    axs = gs.subplots(sharex='col', sharey='row')
#    for ii in range(3):
#        for jj in range(3):
#            insig = amua_uv[cc, ff, dd, ii, jj, :len(t), :]
#            insig = np.flip(insig)
#            insig=filtfilt(bLow,aLow, insig, axis=0, padlen=100)
#            insig=np.abs(insig)
#            insig = np.flip(insig)
#            amua = np.mean(insig, -1)
#            axs[ii, jj].plot(t, amua[:len(t)])
#            axs[ii, jj].set_title('PT_ITD:' + str(stiITD[ii]) + ' ENV_ITD:' + str(stiITD[jj]))
#            axs[ii, jj].set_ylim(3, 30)
#            
#            for ax in fig.get_axes():
#                ax.label_outer()
#    fig.suptitle(fname+'_900pps_0.01s_Channel'+str(cc)+' AMUA trace')
#    axs[2, 1].set_xlabel('Time')
#    fig.text(0.08, 0.5, 'MicroVolt', va='center', rotation='vertical')
#    plt.savefig(results_path+fname+'_Channel'+ str(cc)+'_AMUA', transparent=False)
#    plt.close()
#%%
##%%
#cc = 12
#ff = 0
#dd = 0
#fig,axs = plt.subplots(3,3)
#fig.suptitle(clean_names[0][:-13]+'_chan'+str(cc)+'_'+str(stiRate[ff])+'pps'+'_Dur'+str(stiDur[dd]))
#t = np.arange(0, 0.1, 1/Fs)
#for ii in range(3):
#    for jj in range(3):
#        sig = clean_array[cc, ff, dd, ii, jj, :, :]-np.mean(clean_array[cc, ff, dd, ii, jj, :, :], 0)
##        amua = np.mean(calcAMUA(Fs, sig, Fs_downsample), 1)
#        axs[ii, jj].plot(t, np.mean(sig, 1)[:len(t)])
#        axs[ii, jj].set_title('PT_ITD:' + str(stiITD[ii]) + ' ENV_ITD:' + str(stiITD[jj]))
#        for ax in fig.get_axes():
#            ax.label_outer()
##%%
#reject_idx = np.array(np.where(reject_mark == 1))
#reject_random = random.sample(range(0, reject_idx.shape[1]), 10)
#time_range = 0.01                   
#for x in reject_random:
#    [cc, dd, ii, jj] = reject_idx[:, x]
#    plt.figure(figsize=(15, 10))
#    plt.subplot(1,2,1)
#    plt.plot(AMUA_array[cc, 0, dd, ii, jj, :int(time_range*Fs_down), :])
#    plt.subplot(1,2,2)
#    plt.plot(AMUA_array[cc, 1, dd, ii, jj, :int(time_range*Fs_down), :])
##%%
#accept_idx = np.array(np.where(reject_mark == 0))
#accept_random = random.sample(range(0, accept_idx.shape[1]), 10)
#t = np.arange(0, 0.1, 1/Fs_down)
#Wn = 2*200/Fs_down 
#bLow,aLow = butter(2, Wn, 'lowpass')                  
#for x in accept_random:
#    plt.figure()
#    [cc, dd, ii, jj] = accept_idx[:, x]
#    plt.subplot(1,2,1)
#    insig = AMUA_array[cc, 0, dd, ii, jj, :len(t), :]
#    insig = np.flip(insig)
#    insig=filtfilt(bLow,aLow, insig, axis=0, padlen=100)
#    insig=np.abs(insig)
#    insig = np.flip(insig)
#    amua = np.mean(insig, -1)
#    plt.plot(t, amua)
#    plt.subplot(1,2,2)
#    insig = AMUA_array[cc, 1, dd, ii, jj, :len(t), :]
#    insig = np.flip(insig)
#    insig=filtfilt(bLow,aLow, insig, axis=0, padlen=100)
#    insig=np.abs(insig)
#    insig = np.flip(insig)
#    amua = np.mean(insig, -1)
#    plt.plot(t, amua)























    