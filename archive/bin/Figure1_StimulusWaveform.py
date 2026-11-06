#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:24:26 2022

@author: shiyi
"""

from sys import platform
import numpy as np
import numpy.matlib
import os 
import RZ2ephys as ep
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import as_strided
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, csd, windows, welch, get_window
# give signal path
if platform == "darwin":  
    os.chdir('/Users/fei/Documents/CI_projects/20220127_Analysis')
    fname = '20220127_ENVvsPT_5_P10_1'   
elif platform == "linux":
    Sig_path = '/disks/CIdata/2022_01_11/Results_Data/'
    results_path = '/disks/CIdata/2022_01_27/Results_Data/'
# load stimulus waveform    
if platform == "darwin":
    stimPath = '/Users/fei/Documents/CI_projects/StimData_v1/'
elif platform == "linux":
    stimPath = '/disks/CIdata/1_ENVvsPT/StimuliData/'
StimulusData = np.load(stimPath+'Stim_ENVvsFS_template.npy')
# load stimulus envelope
env_path = '/disks/CIdata/1_ENVvsPT/StimuliData/'
env_name = '20220127_ENVvsPT_5_P5_1_stimEnvdata.npy'
envWave = np.load(env_path+env_name)
# load stimulus parameter
stm_path = '/disks/CIdata/1_ENVvsPT/2_ND/2022_01_27/'
stm = ep.readEphysStimTable(stm_path+'20220127_ENVvsPT_5_P5_1.csv')

Fs = 24414.0625
fname = '20220127_ENVvsPT_5_P5_1_OriginSigArray'
#%%
'''
reorder the random order envWave
'''
ntrial = 15
envWave_array = np.zeros((2, 3, 3, 3, 5000, 2))
stiDur = [0.01, 0.05, 0.2]
stiRate = [900, 4500]
stiITD = [-0.1, 0, 0.1]
stienvITD = [-0.1, 0, 0.1]
for ff in range(2):
    for dd in range(3):
        for ii in range(3):
            for jj in range(3):
                stimParam = [stiRate[ff],stiDur[dd],stiITD[ii],stienvITD[jj]]
                stm_select = stm[(stm['clickRate (Hz)'] == stimParam[0]) & (stm['duration (s)'] == stimParam[1]) & (stm['ITD (ms)'] == stimParam[2]) & (stm['env ITD (ms)'] == stimParam[3])]
                stmIdx = np.array(stm_select.index)
                envWave_array[ff, dd, ii, jj, :, :] = envWave[stmIdx][0, :, :]
#%%
'''
900pps, 0.01s
stimuluts waveform + envelope waveform
figure1_a
'''
from matplotlib.gridspec import GridSpec
ITD = [-0.1, 0, 0.1]
xIdx = [0.305, 0.58, 0.86]
yIdx = [0.81, 0.54, 0.27]
EndIdx = int(Fs*0.011)
hanning = np.hanning(int(0.01*Fs))
t = np.arange(0, 0.011, 1/Fs)*1000
Stimulus = StimulusData[0, 0, :, :, :EndIdx+1, :]
amp = np.max(Stimulus)
Envelope = envWave_array[0, 0, :, :, :EndIdx+1, :]*amp

fig = plt.figure(figsize=(8, 6))
gs = GridSpec(nrows = 3, ncols = 3, wspace=0, hspace=0)
gs.update(left=0.08, right=0.95, bottom = 0.08, top = 0.95, wspace=0.1, hspace=0.1)
for ii in range(3):
    for jj in range(3):
        x = np.abs(ii-2)
        ax = fig.add_subplot(gs[ii, jj])
        ax_xticks = np.arange(1.5, 4, 0.5)
        ax.set_xticks(ax_xticks)
        ax_yticks = np.arange(-140, 140, 50)
        ax.set_yticks(ax_yticks)
#        ax.set_title('PT_ITD'+str(ITD[x])+', ENV_ITD'+str(ITD[jj]))
        ax.plot(t, Stimulus[x, jj, :, 0], color = 'blue', linewidth=2, alpha = 0.8)
        ax.plot(t, Envelope[x, jj, :, 0], color = 'blue', linewidth=2, alpha = 0.5)
        ax.plot(t, Stimulus[x, jj, :, 1], color = 'red', linewidth=2, alpha = 0.8)
        ax.plot(t, Envelope[x, jj, :, 1], color = 'red', linewidth=2, alpha = 0.5)
#        ax.spines['top'].set_visible(False)
#        ax.spines['right'].set_visible(False)
        ax.set_xlim(1.55, 4)
        ax.set_ylim(-20, 160)
        ax.text(1.6, 140, 'PT_ITD:'+str(ITD[x])+', ENV_ITD:'+str(ITD[jj]), fontsize = 12)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        if not ii == 2:
            ax.set_xticks([])
        if not jj == 0:
            ax.set_yticks([])
            
        if ii == 2 and jj == 1:
            ax.set_xlabel('Time (ms)', fontsize = 14)
        if ii == 1 and jj == 0:
            ax.set_ylabel('Stimlus Intensity(dB)', fontsize = 14)
#        ax0 = fig.add_axes([xIdx[ii], yIdx[jj], 0.06, 0.06])
#        ax0.plot(t, Stimulus[x, jj, :, 0], color = 'blue')
#        ax0.plot(t, Envelope[x, jj, :, 0], color = 'blue')
#        ax0.plot(t, Stimulus[x, jj, :, 1], color = 'red')
#        ax0.plot(t, Envelope[x, jj, :, 1], color = 'red') 
#        ax0.set_xlim(0, 3)
#        ax0.set_ylim(-20, 100)
#        ax0.spines['top'].set_visible(False)
#        ax0.spines['right'].set_visible(False)
#        ax0.spines['left'].set_visible(False)
#        ax0.spines['bottom'].set_visible(False)
#        ax0.set_xticks([])
#        ax0.set_yticks([])

plt.savefig('figure1_a.png', transparent = True)

#%%
'''
for NS8001 2023
900pps, 0.01s
stimuluts waveform + envelope waveform
figure1_a
'''
from matplotlib.gridspec import GridSpec
ITD = [-0.12, 0, 0.12]
#ITD = [-0.12, +0.12]
xIdx = [0.305, 0.58, 0.86]
yIdx = [0.81, 0.54, 0.27]
EndIdx = int(Fs*0.011)
hanning = np.hanning(int(0.01*Fs))
t = np.arange(0, 0.011, 1/Fs)*1000
Stimulus = StimulusData[0, 0, :, :, :EndIdx+1, :]
amp = np.max(Stimulus)
Envelope = envWave_array[0, 0, :, :, :EndIdx+1, :]*amp

fig = plt.figure(figsize=(8, 6))
gs = GridSpec(nrows = 2, ncols = 2, wspace=0, hspace=0)
gs.update(left=0.08, right=0.95, bottom = 0.08, top = 0.95, wspace=0.1, hspace=0.1)
for ii in range(2):    
    if ii == 1:
        i = 2
    else:
        i = ii
    for jj in range(2):
        if jj == 1:
            j = 2
        else:
            j = jj
        print(i, j)
        ax = fig.add_subplot(gs[ii, jj])
        ax_xticks = np.arange(1.5, 4, 0.5)
        ax.set_xticks(ax_xticks)
        ax_yticks = np.arange(-140, 140, 50)
        ax.set_yticks(ax_yticks)
#        ax.set_title('PT_ITD'+str(ITD[x])+', ENV_ITD'+str(ITD[jj]))
        ax.plot(t, Stimulus[i, j, :, 0], color = 'blue', linewidth=2, alpha = 0.8)
        ax.plot(t, Envelope[i, j, :, 0], color = 'blue', linewidth=2, alpha = 0.5)
        ax.plot(t, Stimulus[i, j, :, 1], color = 'red', linewidth=2, alpha = 0.8)
        ax.plot(t, Envelope[i, j, :, 1], color = 'red', linewidth=2, alpha = 0.5)
#        ax.spines['top'].set_visible(False)
#        ax.spines['right'].set_visible(False)
        ax.set_xlim(1.55, 4)
        ax.set_ylim(-20, 160)
        ax.text(1.6, 140, 'PT_ITD:'+str(ITD[i])+', ENV_ITD:'+str(ITD[j]), fontsize = 12)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        if not ii == 2:
            ax.set_xticks([])
        if not jj == 0:
            ax.set_yticks([])
            
#        if ii == 2 and jj == 1:
#            ax.set_xlabel('Time (ms)', fontsize = 14)
#        if ii == 1 and jj == 0:
#            ax.set_ylabel('Stimlus Intensity(dB)', fontsize = 14)
#        ax0 = fig.add_axes([xIdx[ii], yIdx[jj], 0.06, 0.06])
#        ax0.plot(t, Stimulus[x, jj, :, 0], color = 'blue')
#        ax0.plot(t, Envelope[x, jj, :, 0], color = 'blue')
#        ax0.plot(t, Stimulus[x, jj, :, 1], color = 'red')
#        ax0.plot(t, Envelope[x, jj, :, 1], color = 'red') 
#        ax0.set_xlim(0, 3)
#        ax0.set_ylim(-20, 100)
#        ax0.spines['top'].set_visible(False)
#        ax0.spines['right'].set_visible(False)
#        ax0.spines['left'].set_visible(False)
#        ax0.spines['bottom'].set_visible(False)
#        ax0.set_xticks([])
#        ax0.set_yticks([])

plt.savefig('figure1_a.png', transparent = True)
#%%
'''
PT_ITD = -0.1, ENV_ITD = 0.1, 0.01s
Figure1_b 900pps &4500pps
'''
EndIdx = int(Fs*0.051)
Stimulus = StimulusData[:, 1, 0, 2, :EndIdx+1, :]
t = np.arange(0, 0.051, 1/Fs)*1000
fig = plt.figure(figsize=(5, 3))
#gs = GridSpec(nrows = 1, ncols = 2)
gs = GridSpec(nrows = 1, ncols = 1)
gs.update(left=0.15, right=0.9, bottom = 0.15, top = 0.9, wspace=0, hspace=0)
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(t, Stimulus[0, :, 0], color = 'blue')
ax0.plot(t, Stimulus[0, :, 1], color = 'red')
ax0.set_title('900pps', fontsize = 14)
ax0.set_xlabel('Time (ms)', fontsize = 14)
ax0.set_ylabel('Stimlus Intensity(dB)', fontsize = 14)
ax0.tick_params(axis='both', which='major', labelsize=14)

#ax1 = fig.add_subplot(gs[0, 1])
#ax1.plot(t, Stimulus[1, :, 0], color = 'blue')
#ax1.plot(t, Stimulus[1, :, 1], color = 'red')
#ax1.set_title('4500pps')
#
#plt.savefig('figure1_b.png', transparent = True)
#%%
'''
PT_ITD = -0.1, ENV_ITD = 0.1, 900pps
Figure1_c 0.01s&0.05s&0.2s
'''
time = [0.011, 0.051, 0.201]
EndIdx = [int(Fs*0.011), int(Fs*0.051), int(Fs*0.201)]
Stimulus = StimulusData[0, :, 0, 2, :, :]
title = ['0.01s', '0.05s', '0.2s']

fig = plt.figure(figsize=(15, 5))
gs = GridSpec(nrows = 3, ncols = 20)

ax = fig.add_subplot(gs[:, :])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
t = np.arange(0, 0.201, 1/Fs)*1000
ax.set_xlim([0, np.max(t)+1])
ax.set_yticks(np.arange(3))
ax.set_yticklabels([0.2, 0.05, 0.01])
ax.set_xlabel('ms')
ax.set_ylabel('Duration Length (s)')

# 0.01s
t0 = np.arange(0, 0.011, 1/Fs)*1000
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(t0, Stimulus[0, :EndIdx[0]+1, 0], color = 'blue')
ax0.plot(t0, Stimulus[0, :EndIdx[0]+1, 1], color = 'red')
ax0.set_title('0.01s')
ax0.set_xlim([0, np.max(t0)+1])
ax0.set_ylim([np.min(Stimulus[0, :EndIdx[0]+1, 0])-10, np.max(Stimulus[0, :EndIdx[0]+1, 0])+10])
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['left'].set_visible(False)
ax0.spines['bottom'].set_visible(False)
ax0.set_xticks([])
ax0.set_yticks([])

# 0.05s
t1 = np.arange(0, 0.051, 1/Fs)*1000
ax1 = fig.add_subplot(gs[1, :6])
ax1.plot(t1, Stimulus[1, :EndIdx[1]+1, 0], color = 'blue')
ax1.plot(t1, Stimulus[1, :EndIdx[1]+1, 1], color = 'red')
ax1.set_title('0.05s')
ax1.set_xlim([0, np.max(t1)+1])
ax1.set_ylim([np.min(Stimulus[1, :EndIdx[1]+1, 0])-10, np.max(Stimulus[1, :EndIdx[1]+1, 0])+10])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.set_xticks([])
ax1.set_yticks([])

# 0.2s
t2 = np.arange(0, 0.201, 1/Fs)*1000
ax2 = fig.add_subplot(gs[2, :])
ax2.plot(t2, Stimulus[2, :EndIdx[2]+1, 0], color = 'blue')
ax2.plot(t2, Stimulus[2, :EndIdx[2]+1, 1], color = 'red')
ax2.set_title('0.2s')
ax2.set_xlim([0, np.max(t2)+1])
ax2.set_ylim([np.min(Stimulus[2, :EndIdx[2]+1, 0])-10, np.max(Stimulus[2, :EndIdx[2]+1, 0])+10])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.set_xticks([])
ax2.set_yticks([])

plt.savefig('figure1_c.png', transparent = True)
#%%
'''
900pps, 0.01s
stimuluts waveform + envelope waveform
'''
EndIdx = int(Fs*0.011)
hanning = np.hanning(int(0.01*Fs))
t = np.arange(0, 0.011, 1/Fs)*1000
Stimulus = StimulusData[0, 0, :, :, :EndIdx+1, :]
amp = np.max(Stimulus)
Envelope = envWave_array[0, 0, :, :, :EndIdx+1, :]*amp
plt.figure(figsize=(15, 10))
plt.subplot(3,3,1)
plt.title('PT_ITD0.1, ENV_ITD-0.1')
plt.plot(t, Stimulus[2, 0, :, 0], color = 'blue')
plt.plot(t, Envelope[2, 0, :, 0], color = 'blue')
plt.plot(t, Stimulus[2, 0, :, 1], color = 'red')
plt.plot(t, Envelope[2, 0, :, 1], color = 'red')
plt.xlim(0, 5)
plt.ylim(-20, 170)
plt.subplot(3,3,2)
plt.title('PT_ITD0.1, ENV_ITD0')
plt.plot(t, Stimulus[2, 1, :, 0], color = 'blue')
plt.plot(t, Envelope[2, 1, :, 0], color = 'blue')
plt.plot(t, Stimulus[2, 1, :, 1], color = 'red')
plt.plot(t, Envelope[2, 1, :, 1], color = 'red')
plt.xlim(0, 5)
plt.ylim(-20, 170)
plt.subplot(3,3,3)
plt.title('PT_ITD0.1, ENV_ITD0.1')
plt.plot(t, Stimulus[2, 2, :, 0], color = 'blue')
plt.plot(t, Envelope[2, 2, :, 0], color = 'blue')
plt.plot(t, Stimulus[2, 2, :, 1], color = 'red')
plt.plot(t, Envelope[2, 2, :, 1], color = 'red')
plt.xlim(0, 5)
plt.ylim(-20, 170)
plt.subplot(3,3,4)
plt.title('PT_ITD0, ENV_ITD-0.1')
plt.plot(t, Stimulus[1, 0, :, 0], color = 'blue')
plt.plot(t, Envelope[1, 0, :, 0], color = 'blue')
plt.plot(t, Stimulus[1, 0, :, 1], color = 'red')
plt.plot(t, Envelope[1, 0, :, 1], color = 'red')
plt.xlim(0, 5)
plt.ylim(-20, 170)
plt.subplot(3,3,5)
plt.title('PT_ITD0, ENV_ITD0')
plt.plot(t, Stimulus[1, 1, :, 0], color = 'blue')
plt.plot(t, Envelope[1, 1, :, 0], color = 'blue')
plt.plot(t, Stimulus[1, 1, :, 1], color = 'red')
plt.plot(t, Envelope[1, 1, :, 1], color = 'red')
plt.xlim(0, 5)
plt.ylim(-20, 170)
plt.subplot(3,3,6)
plt.title('PT_ITD0, ENV_ITD0.1')
plt.plot(t, Stimulus[1, 2, :, 0], color = 'blue')
plt.plot(t, Envelope[1, 2, :, 0], color = 'blue')
plt.plot(t, Stimulus[1, 2, :, 1], color = 'red')
plt.plot(t, Envelope[1, 2, :, 1], color = 'red')
plt.xlim(0, 5)
plt.ylim(-20, 170)
plt.subplot(3,3,7)
plt.title('PT_ITD-0.1, ENV_ITD-0.1')
plt.plot(t, Stimulus[0, 0, :, 0], color = 'blue')
plt.plot(t, Envelope[0, 0, :, 0], color = 'blue')
plt.plot(t, Stimulus[0, 0, :, 1], color = 'red')
plt.plot(t, Envelope[0, 0, :, 1], color = 'red')
plt.xlim(0, 5)
plt.ylim(-20, 170)
plt.subplot(3,3,8)
plt.title('PT_ITD-0.1, ENV_ITD0')
plt.plot(t, Stimulus[0, 1, :, 0], color = 'blue')
plt.plot(t, Envelope[0, 1, :, 0], color = 'blue')
plt.plot(t, Stimulus[0, 1, :, 1], color = 'red')
plt.plot(t, Envelope[0, 1, :, 1], color = 'red')
plt.xlim(0, 5)
plt.ylim(-20, 170)
plt.subplot(3,3,9)
plt.title('PT_ITD-0.1, ENV_ITD0.1')
plt.plot(t, Stimulus[0, 2, :, 0], color = 'blue')
plt.plot(t, Envelope[0, 2, :, 0], color = 'blue')
plt.plot(t, Stimulus[0, 2, :, 1], color = 'red')
plt.plot(t, Envelope[0, 2, :, 1], color = 'red')
plt.xlim(0, 5)
plt.ylim(-20, 170)

#%%
'''
ch26, 900pps/4500pps, 0.01s 
take first trail to give an example of artifact shape
'''
Sig_path = '/disks/CIdata/2022_01_27/Results_Data/'
origin_name = '20220127_ENVvsPT_5_P5_1_OriginSigArray.npy'
origin_array = np.load(Sig_path+origin_name)
t = np.arange(0, 0.011, 1/Fs)*1000 # time unit: ms
EndIdx = int(Fs*0.011)
stiDur = [0.01, 0.05, 0.2]
stiITD = [-0.1, 0, 0.1]
ff = 0# 0:900pps, 1: 4500pps
dd = 0
cc = 26
origin = origin_array[cc, ff, dd, :, :, :, :] 
plt.figure(figsize=(15, 10))
for ii in range(3):
    for jj in range(3):
        x = ii*3+jj+1
        plt.subplot(3,3,x)
#        plt.xlim(5.2, 6.7)
        z = np.abs(ii-2)
        plt.plot(t, origin[z, jj, 1:EndIdx+2, 0]-origin[z, jj, 1, 0])
        plt.title('PT_ITD ' + str(stiITD[z]) + ' ENV_ITD ' + str(stiITD[jj]))




















