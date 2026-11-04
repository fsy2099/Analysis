#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 17:08:50 2022

@author: shiyi
"""

from sys import platform
import numpy as np
import numpy.matlib
import os 
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import as_strided
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, csd, windows, welch, get_window, resample
import random
from matplotlib.gridspec import GridSpec
from scipy import stats
import seaborn as sns 
from math import fmod

#Sig_path = '/disks/CIdata/1_ENVvsPT/1_NH/2021_12_16/Results_Data/Average/'
Sig_path = '/disks/CIdata/1_ENVvsPT/1_NH/2022_03_02/Results_Data/'
#results_path = '/disks/CIdata/1_ENVvsPT/1_NH/2021_12_16/Results_Data/Average/'
results_path = '/disks/CIdata/1_ENVvsPT/1_NH/2022_03_02/Results_Data/'
selction_path = '/disks/CIdata/1_ENVvsPT/selection/ND/_AMUA_Selection/'
#%%
stiDur = [0.01, 0.05, 0.2]
stiRate = [900, 4500]
stiITD = [-0.1, 0, 0.1]
stienvITD = [-0.1, 0, 0.1]
Fs = 24414.0625
Fs_down = 2000
nsamples_down = int(Fs_down*0.5)-1 

file_names = os.listdir(Sig_path)
sig_names = [file_name for file_name in file_names if all([x in file_name for x in [ "_OriginSigArray_AMUA_Mean.npy"]])]
#sig_names = [file_name for file_name in file_names if all([x in file_name for x in [ "_2R_BI_1_AMUA.npy"]])]
print(sig_names)
#%% plot AMUA trace and raw clean signal (900pps, 0.05s)for each channel
'''
applied 200Hz low pass filter to AMUA
3k highpass to clean array
if the peak response of AMUA within time window 0~0.05s > average baseline + 5*std
the channel labeled as 'reponsed'
'''
stiDur = [0.01, 0.05, 0.2]
ff = 1
dd = 1
#t_start = 0.4 # time window of baseline
#t_end = 0.5
#idx_start = round(t_start*Fs_down) 
#idx_end = round(t_end*Fs_down)-1
t = np.arange(0, 0.2, 1/Fs_down)
T = np.arange(0, 0.2, 1/Fs)
dash = np.repeat(stiDur[dd], 40)
Wn = 2*200/Fs_down
nyq = 0.5*Fs
bLow,aLow = butter(2, Wn, 'lowpass')
bhigh2,ahigh2 = butter(2, 3000/nyq, 'highpass')

for sig_name in sig_names:
#sig_name = sig_names[0]
    print(sig_name)
    amua_array = np.load(Sig_path+sig_name, allow_pickle = True)
#    clean_name = sig_name[:-9] + '_OriginSigArray_CleanSig.npy'
    clean_name = sig_name[:-14] + '_CleanSig_Mean.npy'
    clean_array = np.load(Sig_path+clean_name, allow_pickle = True)
#    selection_name = sig_name[:-9] + '_AMUA_Selection.npy'
#    selection_name = sig_name[:-14] + '_AMUA_Selection.npy'
#    selection_array = np.load(selction_path+selection_name, allow_pickle = True)
    
    for cc in range(32):          
        fig = plt.figure(figsize=(20, 10))
        fig.set_tight_layout(True)
        gs = GridSpec(nrows = 3, ncols = 6)
        plt.suptitle(sig_name[:-9]+'_Ch'+str(cc))       

        for ii in range(3):
            for jj in range(3):
                # apply 200Hz low pass filter on AMUA
                amua = amua_array[cc, ff, dd, ii, jj, :]
                amua = filtfilt(bLow,aLow, amua, axis=0, padlen=100)
                amua = np.mean(amua, -1)
                
                ax = fig.add_subplot(gs[ii, jj])
                ax.set_title('PT_ITD: '+str(stiITD[ii])+' ENV_ITD: '+str(stienvITD[jj]))
                ax.plot(t, amua[:len(t)]*1000000)
                ax.plot(dash, np.arange(40), 'r--')
                ax.set_ylim(2, 35)
                
                clean = clean_array[cc, ff, dd, ii, jj, 1:, :]
                clean=filtfilt(bhigh2,ahigh2, clean, axis=0, padlen=100)
                clean = clean[:len(T), :]
                
                ax = fig.add_subplot(gs[ii, jj+3])   
                ax.set_title('PT_ITD: '+str(stiITD[ii])+' ENV_ITD: '+str(stienvITD[jj]))
                ax.plot(T, clean*1000000)
#        if selection_array[cc] == 1:
#            plt.suptitle(sig_name[:-9]+'_Ch'+str(cc), color= 'red')       
        
        plt.savefig(results_path+sig_name[:-9]+'_Channel'+ str(cc)+'_AMUA')
        plt.close()

#    np.save(selction_path+sig_name[:-9]+'_AMUA_Selection.npy',selection_array)

#%% plot 900pps, 0.05s, AMUA heatmap
'''
average AMUA over time range (0 to 0.05s)
use Z-score normalize the average AMUA 
plot heatmap by z-score
'''    
ANOVA_path = '/disks/CIdata/1_ENVvsPT/ANOVA/NH/'
AMUA_path = '/disks/CIdata/1_ENVvsPT/AMUA/NH/'
selection_path = '/disks/CIdata/1_ENVvsPT/selection/NH/_AMUA_Selection/'
results_path = '/disks/CIdata/1_ENVvsPT/AMUA/Heatmap/'

ff = 0
dd = 1
t_end = 0.05
IdxEnd = round(t_end*Fs_down)

file_names = os.listdir(selection_path)
selection_names = [file_name for file_name in file_names if all([x in file_name for x in [ 'P1', '20220822', "_AMUA_Selection.npy"]])]
print(selection_names)

for selection_name in selection_names:
#selection_name = selection_names[2]  
    selection_array = np.load(selection_path+selection_name)
    sig_name = selection_name[:-14]+'.npy'
    print(sig_name)
    amua_array = np.load(AMUA_path+sig_name, allow_pickle = True)
    VE_name = sig_name[:-4]+'_ExplainedVariance.npy'
    print(VE_name)
    VE_array = np.load(ANOVA_path+VE_name)    # VE_array[0, :] is VEpt; VE_array[1, :] is VEenv
    
    amua = np.mean(np.mean(amua_array[:, ff, dd, :, :, 1:IdxEnd+1, :], -1), -1)
    amua_z = np.zeros((32, 3, 3))
    
    for cc in range(32):
        amua_z[cc, :, :] = stats.zscore(amua[cc, :, :], axis = None)    
    
    fig = plt.figure(figsize=(20, 10))
    fig.set_tight_layout(True)
    gs = GridSpec(nrows = 4, ncols = 8)
    plt.suptitle(sig_name[:-9]+'_AMUA')
    
    for cc in range(32):
        x = cc%8
        y = cc//8
    
        ax = fig.add_subplot(gs[y, x])
        sns.heatmap(amua_z[cc, :, :], ax=ax, vmin=np.min(amua_z), vmax=np.max(amua_z), cmap='plasma')
        if selection_array[cc] == 1:            
            ax.set_title('Ch'+str(cc)+' VEpt:'+str('%0.4f' %(VE_array[cc, ff, dd, 0]*100))+'\n'+'VEenv: '+str('%0.4f' %(VE_array[cc, ff, dd, 1]*100)), color = 'red')
        elif selection_array[cc] == 0:
            ax.set_title('Ch'+str(cc)+' VEpt:'+str('%0.4f' %(VE_array[cc, ff, dd, 0]*100))+'\n'+'VEenv: ' + str('%0.4f' %(VE_array[cc, ff, dd, 1]*100)))
    #    plt.xticks(np.arange(3), ['-0.1', '0', '0.1'])
    #    plt.yticks(np.arange(3), ['-0.1', '0', '0.1'])
    plt.savefig(results_path+sig_name[:-9]+'_HeatMap') 
    plt.close()    
                                   #%%
### plot amua trace with 900pps, 0.01s
##ff = 0
##dd = 0
##for cc in range(32):
##    fig = plt.figure(figsize=(10,15), frameon = False)
##    gs = fig.add_gridspec(3, 3, hspace=0.25, wspace=0.15)
##    axs = gs.subplots(sharex='col', sharey='row')
##    for ii in range(3):
##        for jj in range(3):
##            insig = amua_uv[cc, ff, dd, ii, jj, :len(t), :]
##            insig = np.flip(insig)
##            insig=filtfilt(bLow,aLow, insig, axis=0, padlen=100)
##            insig=np.abs(insig)
##            insig = np.flip(insig)
##            amua = np.mean(insig, -1)
##            axs[ii, jj].plot(t, amua[:len(t)])
##            axs[ii, jj].set_title('PT_ITD:' + str(stiITD[ii]) + ' ENV_ITD:' + str(stiITD[jj]))
##            axs[ii, jj].set_ylim(3, 30)
##            
##            for ax in fig.get_axes():
##                ax.label_outer()
##    fig.suptitle(fname+'_900pps_0.01s_Channel'+str(cc)+' AMUA trace')
##    axs[2, 1].set_xlabel('Time')
##    fig.text(0.08, 0.5, 'MicroVolt', va='center', rotation='vertical')
##    plt.savefig(results_path+fname+'_Channel'+ str(cc)+'_AMUA', transparent=False)
##    plt.close()
##%% 
#'''
#plot heatmap
#'''
#position = 'P3'
#file_names = os.listdir(Sig_path)
##for pp in position:    
#sig_name = [file_name for file_name in file_names if all([x in file_name for x in [position, "_OriginSigArray_AMUA.npy"]])]
#amua_array = np.load(Sig_path+sig_name, allow_pickle = True)
#fname = sig_name[:-24]
#selection_path = '/disks/CIdata/selection/ND/'
#file_names = os.listdir(Sig_path)
##    for x in sig_names:
##        amua_array = np.load(Sig_path+x, allow_pickle = True)
##        fname = x[:-24]
##%%
#t = np.arange(0.005, 0.04, 1/Fs_down)
#IdxStart = round(0.005*Fs_down)
#IdxEnd = round(0.04*Fs_down)
#channel = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
#ff = 0
#dd = 0
#stiRate = [900, 4500]
#stiDur = [0.01, 0.05, 0.2]
#stiITD = [-0.1, 0, 0.1]
#amua_plot = np.mean(np.mean(amua_array[channel, ff, dd, :, :, :IdxEnd, :], -1), -1)
#amua_z = np.zeros((len(channel), 3, 3))
#for x in range(len(channel)):
#    amua_z[x, :, :] = stats.zscore(amua_plot[x, :, :])
##amua_z = stats.zscore(amua_plot, axis=1)
##%%
#for cc in range(len(channel)):
##cc = 0
#    plt.figure(figsize=(4, 4), frameon = False)
#    #gs = fig.add_gridspec(2, 5, hspace=0.2, wspace=0.1)
#    sns.heatmap(amua_z[cc, :, :],cmap='plasma',cbar=False)
#    plt.xticks(np.arange(3), ['-0.1', '0', '0.1'])
#    plt.yticks(np.arange(3), ['-0.1', '0', '0.1'])
#    plt.savefig(Sig_path+fname+'_channel'+str(cc)+'_HeatMap', transparent=False)
#    plt.close()
##axs = gs.subplots(sharex='col', sharey='row')
##fig.suptitle('ND_900pps_0.01s AMUA')
##cbar_ax = fig.add_axes([.91, .3, .03, .5])
##for i, ax in enumerate(axs.flat):
##    sns.heatmap(amua_z[i, :, :], ax=ax, cbar=i == 0, vmin=-1.5, vmax=1.5, cmap='plasma',
##                cbar_ax=None if i else cbar_ax)
##    ax.set_title('channel' + str(channel[i]))
##    ax.set_xticklabels(['-0.1', '0', '0.1'])
##    ax.set_yticklabels(['-0.1', '0', '0.1'])
##axs[1, 2].set_xlabel('ENV_ITD')
##fig.text(0.1, 0.5, 'PT_ITD', va='center', rotation='vertical')
##%%
#fig = plt.figure()
#gs = fig.add_gridspec(1, 7, hspace=0.1, wspace=0.01)
#axs = gs.subplots(sharex='col', sharey='row')
#fig.suptitle('ND_900pps_0.01s AMUA')
#for x in range(6):
#        g = sns.heatmap(amua_plot[x, :, :],cmap='plasma',cbar=False,ax=axs[x])
#        axs[x].set_xticklabels(['-0.1', '0', '0.1'])
#        axs[x].set_title('channel' + str(channel[x]))
#        axs[x].set_aspect('equal')
#g = sns.heatmap(amua_plot[6, :, :],cmap='plasma',ax=axs[6])
#axs[6].set_aspect('equal')
#axs[6].set_title('channel' + str(channel[6]))
#axs[6].set_xticklabels(['-0.1', '0', '0.1'])
#axs[0].set_yticklabels(['-0.1', '0', '0.1'])
#axs[3].set_xlabel('ENV_ITD')
#axs[0].set_ylabel('PT_ITD')
##%%
#fig = plt.figure(figsize=(15,7), frameon = False)
#gs = fig.add_gridspec(2, 5, hspace=0.2, wspace=0.1)
#axs = gs.subplots(sharex='col', sharey='row')
#fig.suptitle('ND_900pps_0.01s AMUA')
#cbar_ax = fig.add_axes([.91, .3, .03, .5])
#for i, ax in enumerate(axs.flat):
#    sns.heatmap(amua_z[i, :, :], ax=ax, cbar=i == 0, vmin=-1.5, vmax=1.5, cmap='plasma',
#                cbar_ax=None if i else cbar_ax)
#    ax.set_title('channel' + str(channel[i]))
#    ax.set_xticklabels(['-0.1', '0', '0.1'])
#    ax.set_yticklabels(['-0.1', '0', '0.1'])
#axs[1, 2].set_xlabel('ENV_ITD')
#fig.text(0.1, 0.5, 'PT_ITD', va='center', rotation='vertical')
##    ax.set_aspect('equal')
##for x in range(6):
##        g = sns.heatmap(amua_z[x, :, :],cmap='plasma',cbar=False,ax=axs[x])
##        axs[x].set_xticklabels(['-0.1', '0', '0.1'])
##        axs[x].set_title('channel' + str(channel[x]))
##g = sns.heatmap(amua_z[6, :, :],cmap='plasma',ax=axs[6])
##axs[6].set_title('channel' + str(channel[6]))
##axs[6].set_xticklabels(['-0.1', '0', '0.1'])
##axs[0].set_yticklabels(['-0.1', '0', '0.1'])
##axs[3].set_xlabel('ENV_ITD')
##axs[0].set_ylabel('PT_ITD')