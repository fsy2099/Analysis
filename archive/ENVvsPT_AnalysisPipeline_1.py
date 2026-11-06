#!/usr/bin/env python3  
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:37:31 2024

@author: jan
"""

import glob
import os
import pandas as pd 
import matplotlib.pyplot as plt
from sys import path
path.append('/home/colliculus/behaviourBoxes/software/ratCageProgramsV3/')
import RZ2ephys as ep
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
import statistics
import re
 
dataRoot='/ephys/2/0_ENVvsPT/Data'
analysisRoot='/ephys/2/0_ENVvsPT/Analysis'
figureRoot = '/ephys/2/0_ENVvsPT/Figure'
#%%
'''
Read in & write in
'''
def readEphysFileFromRow(dataset, row, loadOrig = False, loadAMUA = False, loadClean = False):
    if loadOrig == True:
        filename = dataset.fileName[row]
    if loadAMUA == True:
        filename = dataset.fileName[row][:-6]+'_AMUA'
    if loadClean == True:
        filename = dataset.fileName[row][:-6]+'_Clean'
    if not any('Cleaned' in col for col in dataset.columns):# if no 'Cleaned' column in csv file, read in all row
        swps, stimPar=ep.readEphysFile(dataRoot+'/'+dataset.animDir[row]+'/'+filename)
        swps, stimPar = checkSwpsSize(swps, stimPar)
        print(filename)
    if any('Cleaned' in col for col in dataset.columns):# if exist 'Cleaned' column in csv file, only read in row labed as '1' in cleaned
        if dataset.Cleaned[row] == 1:
            swps, stimPar=ep.readEphysFile(dataRoot+'/'+dataset.animDir[row]+'/'+filename)
            swps, stimPar = checkSwpsSize(swps, stimPar)
            print(filename)
        if dataset.Cleaned[row] == 0:
            swps = []
            stimPar = []
    return swps, stimPar

def writeEphysFileFromRow(dataset, row, swps, stimPar, writeAMUA = False, writeClean = False):
    if writeAMUA == True:
        filename = dataset.fileName[row][:-6]+'_AMUA'
    if writeClean == True:
        filename = dataset.fileName[row][:-6]+'_Clean'
    ep.writeEphysFile(dataRoot+'/'+dataset.animDir[row]+'/'+filename, swps, stim = stimPar)

def writeCsvFileFromRow(dataset, row, writeWilcoxon = False):
    if writeWilcoxon == True:
        filename = dataset.fileName[row][:-6]+'.csv'
    dataset.to_csv(dataRoot+'/'+dataset.animDir[row]+'/'+filename, index=False)
    
def checkSwpsSize(swps, stimPar):
    # for HK data, some set recorded by 64 channel electrode containing IC and AC, only take first 32ch(IC)
    # for FR data, some set last swp is missed or shorter than normal, replace by second last swp
    if swps[1].signal.shape[1] != 32:
        for swp in swps:
            swp.signal = swp.signal[:, :32]
    if len(swps) == len(stimPar):
        # If last swp not recorded
        Last_conds = findLastCond(stimPar)
        Last_swpIdx=findSweepsFor(Last_conds, stimPar)
        swps = np.append(swps, swps[Last_swpIdx[-2]])
        
    # Code by Tim to get rid of the fact that sometimes the one sweep was too short (2* stim duration in sec):
    # Can also adjust the duration sweeps has to have 
    Short_swps = [] #List of short swps, adjusted by the for loop below... 
    for idx in  range (len(stimPar)):
        if swps[idx+1].signal.shape[0]/swps[-1].sampleRate < 2*(float(stimPar.iloc[idx]['duration (s)'])):
            Short_swps.append(idx)
            print('sweeps too short:', idx)
    
    for idx, Short_swp in enumerate (Short_swps):
        Last_duration = float(stimPar.iloc[Short_swp]['duration (s)'])
        Last_clickRate = int(stimPar.iloc[Short_swp]['clickRate (Hz)'])
        Last_ITD = float(stimPar.iloc[Short_swp]['ITD (ms)'])
        Last_Env = float(stimPar.iloc[Short_swp]['env ITD (ms)'])
        Last_conds = [('duration (s)',Last_duration), ('clickRate (Hz)',Last_clickRate), ('ITD (ms)',Last_ITD), ('env ITD (ms)',Last_Env)]
        Last_swpIdxs=findSweepsFor(Last_conds, stimPar)
        for i, Last_swpIdx in enumerate (Last_swpIdxs):
            if Last_swpIdx == Short_swp+1:
                try:
                    swps[Last_swpIdxs[i]] = swps[Last_swpIdxs[i - 1]]
                except: 
                    swps[Last_swpIdxs[i]] = swps[Last_swpIdxs[i + 1]]
    return swps, stimPar    

def findLastCond(stimPar):
    Last_duration = float(stimPar.iloc[-1]['duration (s)'])
    Last_clickRate = int(stimPar.iloc[-1]['clickRate (Hz)'])
    Last_ITD = float(stimPar.iloc[-1]['ITD (ms)'])
    Last_Env = float(stimPar.iloc[-1]['env ITD (ms)'])
    Last_conds = [('duration (s)',Last_duration), ('clickRate (Hz)',Last_clickRate), ('ITD (ms)',Last_ITD), ('env ITD (ms)',Last_Env)]
    return Last_conds
#%%
'''
some useful function
'''
def findSweepsFor(conditions, stimPar):
    # returns a list of indeces into for the sweeps that satisfy the conditions.
    # conditions should be a list,  of pairs (paramName, paramValue)
    # For now e only do equality, and logical AND, ie we look for sweeps
    # satisfying (paramName1==paramValue1) AND (paramName2==paramValue2) ...    
    if not type(conditions)==list:
        raise Exception("conditions must be a list of pairs, eg [('duration (s)',0.01),('clickRate (Hz)',900)]")
    acond=conditions.pop()
    select=stimPar[acond[0]]==acond[1]
    while len(conditions)>0:
        acond=conditions.pop()
        select=np.logical_and(select,stimPar[acond[0]]==acond[1])
    swpIdx=np.where(select)[0]      
    return swpIdx + 1  

def findConditionsForswp (swp, stimPar):
    duration = float(stimPar.iloc[swp-1]['duration (s)'])
    clickRate = int(stimPar.iloc[swp-1]['clickRate (Hz)'])
    ITD = float(stimPar.iloc[swp-1]['ITD (ms)'])
    Env = float(stimPar.iloc[swp-1]['env ITD (ms)'])
    conds = [('duration (s)',duration), ('clickRate (Hz)',clickRate), ('ITD (ms)',ITD), ('env ITD (ms)',Env)]
    return (conds)

def poolChannel(swps, channelNum):
    # pool out all repeats for the given channel
    alen = np.zeros([len(swps)])
    for ss in range(len(swps)): 
        alen[ss]=swps[ss].signal.shape[0]
    minLen=int(np.min(alen))
    signal = []
    for swp in swps:
        signal.append(swp.signal[:minLen, channelNum])
    signal = np.array(signal).T
    return signal

def findRecordingSite(dataFiles, row):
    filename = dataFiles['fileName'][row]
    sub_recordingSites = recordingSites[recordingSites['fileName']==filename]
    sub_recordingSites = sub_recordingSites[sub_recordingSites['Responsive'] == 1]     
    return sub_recordingSites

def writeValueinto(df, idx, V1, V2, Vp1, Vp2, KeyName):
    df = checkKeyNamein(df, keyName)
    df.at[idx, keyName+'_VEpt'] = V1
    df.at[idx, keyName+'_VEenv'] = V2                     
    df.at[idx, '(pValue)'+keyName+'_VEpt'] = Vp1
    df.at[idx, '(pValue)'+keyName+'_VEenv'] = Vp2
    if V1>Vp1:
        df.at[sub_recordingSites.index[channel], keyName+'_VEpt_significance'] = 1
    else:
        df.at[sub_recordingSites.index[channel], keyName+'_VEpt_significance'] = 0
    if V2>Vp2:
        df.at[sub_recordingSites.index[channel], keyName+'_VEenv_significance'] = 1
    else:
        df.at[sub_recordingSites.index[channel], keyName+'_VEenv_significance'] = 0
    return df


def checkKeyNamein(recordingSites, keyName):
    if not any(keyName in col for col in recordingSites.columns):
        recordingSites[keyName+'_VEpt'] = ''
        recordingSites['(pValue)'+keyName+'_VEpt'] = ''
        recordingSites[keyName+'_VEpt_significance'] = ''
        recordingSites[keyName+'_VEenv'] = ''
        recordingSites['(pValue)'+keyName+'_VEenv'] = ''
        recordingSites[keyName+'_VEenv_significance'] = ''
    return recordingSites
#%%
"""  
Artifact Rejection &
AMUA Calculation
    
"""
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, csd, windows, welch, get_window, resample, lfilter


def AMUAFilterCoeffs(fs,lowpass=6000):
        nyq = 0.5*fs
        bBand,aBand = butter(2,(300/nyq, 6000/nyq),'bandpass')
        bLow,aLow = butter(2,(lowpass/nyq),'lowpass')
        bNotch, aNotch = iirnotch(50, 30, fs)
        return [[bBand, aBand], [bLow, aLow], [bNotch, aNotch]]
    
def calcAMUA(swp, Fs_downsample, padLen=300):
# def calcAMUA(fs, ori_signal, Fs_downsample, padLen=300):
        fs=swp.sampleRate
        coefs=AMUAFilterCoeffs(fs)
        bpCoefs=coefs[0]
        lpCoefs=coefs[1]
        NotchCoefs = coefs[2]
        insig = filtfilt(NotchCoefs[0], NotchCoefs[1], swp.signal, axis=0, padlen=padLen)
        insig = np.flip(insig)
        insig=filtfilt(bpCoefs[0],bpCoefs[1], insig, axis=0, padlen=padLen)
        insig=np.abs(insig)
        insig=filtfilt(lpCoefs[0],lpCoefs[1],insig,axis=0, padlen=padLen)
        insig = np.flip(insig)          
        # Fs_downsample
        # signal = resample_poly(insig, Fs_downsample, int(fs), axis=0)
        downsample_length = int((insig.shape[0]/fs)*Fs_downsample)
        signal=resample(insig,downsample_length)
        result=ep.MCsweep(Fs_downsample)
        result.timeStamp=swp.timeStamp
        result.signal=signal
        return result
    
def remove_jump (swps, stimPar):
    '''
    Parameters
    ----------
    swp : tAKE in several sweeps as MCephys object.

    Returns: A cleaned version of the sweep in case jumps happen that are bigger than 5*std
    -------
    swp : TYPE
        DESCRIPTION.
    '''
    sampleRate = swps[1].sampleRate
    durations = stimPar["duration (s)"].values

    for i in range(len(swps)):
        if i == 0:
            continue
        # print (i)
        for cc in range (32):
            differences = np.abs(np.diff(swps[i].signal[int(durations[i-1]*sampleRate):,cc], n = 5))
            threshold = np.mean(differences) + 10 * np.std(differences)  #adjust here the criterion to find the jumps, right now its 5times std...
            jump_indices = np.where(differences > threshold)[0] + 1 # get the indeces where jumps are 
            jump_indices = jump_indices + int(durations[i-1]*sampleRate)
            for jump_index in jump_indices:
                # print(idx)
                print("JUMP DETECTED in" + str(i))
                shift = swps[i].signal[jump_index -1 ,cc] - swps[i].signal[jump_index ,cc]
                swps[i].signal[jump_index:,cc] += shift
    return swps

#%%
'''
Wilcoxon Test
'''
def wilcoxonTestfor(swps, stimPar, conditions):
    swpIdx=findSweepsFor(conditions.copy(), stimPar)
    swpTest = swps[swpIdx]
    sampleRate = swpTest[0].sampleRate
    nchans = swpTest[0].signal.shape[1]
    duration = conditions[0][1]
    selectionArray = []
    resultsArray = []
    for cc in range(nchans):
        signal = poolChannel(swpTest, cc)
        responseWinAvg = signal[:int(duration*sampleRate), :].mean(axis = 0)
        baselineWinAvg = signal[-int(duration*sampleRate):, :].mean(axis = 0)
        results = stats.wilcoxon(responseWinAvg, baselineWinAvg)
        resultsArray.append(results[1])
        if results[1] < 0.05:
            selectionArray.append(1)
        else:
            selectionArray.append(0)
    return resultsArray, selectionArray

#%% 
'''
variance explained ratio calculation     
'''   
def calcVarianceExplained(df):
    ss_t = ssTotalCalc(df)
    model = ols('AMUA ~ C(PT_ITD)+C(ENV_ITD)+C(PT_ITD):C(ENV_ITD)', data = df).fit()
    result = sm.stats.anova_lm(model, type=1)
    ss_pt = result.iloc[0,1]/ss_t
    ss_env = result.iloc[1,1]/ss_t
    return ss_pt, ss_env

def ssTotalCalc(df):
    totalMean = df['AMUA'].mean()
    ss_t = sum((df['AMUA']-totalMean)**2)
    return ss_t

def permutationTestfor(df, num=1000):
    ss_pt_array = []
    ss_env_array = []
    for tt in range(num):
        shuffled_index = np.random.permutation(df.index)
        df['AMUA'] = df['AMUA'].iloc[shuffled_index].values
        ss_pt, ss_env = calcVarianceExplained(df)
        ss_pt_array.append(ss_pt)
        ss_env_array.append(ss_env)
    ss_pt_p = np.percentile(np.array(ss_pt_array), 95)
    ss_env_p = np.percentile(np.array(ss_env_array), 95)
    return ss_pt_p, ss_env_p
#%%
'''
Visulization tool
'''
def prepare4X8Plot():
    fig = plt.figure(figsize = (20, 10))
    fig.set_tight_layout(True)
    gs = GridSpec(4, 8)
    axs = []
    for ii in range(32):
        x = ii%8
        y = ii//8        
        ax = fig.add_subplot(gs[y, x])
        ax.set_title('Ch'+str(ii))
        axs.append(ax)
    return fig, axs

def plot32Channel(swps, conditions, allRepeats = False, avgRepeats = False, ncols = 8, nrows = 4):
    sampleRate = swps[1].sampleRate
    Duration = conditions[0][1]
    t = np.arange(0, Duration+0.05, 1/sampleRate)
    nchans = swps[1].signal.shape[1]
    swpIdx=findSweepsFor(conditions.copy(), stimPar)
    swpPlot = swps[swpIdx]
    
    fig, axs = prepare4X8Plot()
    for cc in range(nchans):
        ax = axs[cc]        
        signal = poolChannel(swpPlot, cc)
        if avgRepeats:
            signal = signal[:len(t), :].mean(axis = 1)
            ax.plot(t, signal)
        if allRepeats:
            signal = (signal-signal[0, :])[:len(t)]
            ax.plot(t, signal)

        ax.axvline(x = Duration, color = 'k', linestyle='--')
        if not np.mod(cc,ncols) == 0:
            plt.yticks([])
        if not cc>=nchans-ncols:
            plt.xticks([])
    return fig

def prepare3X3Plot(itd = [-0.1, 0, 0.1]):   
    fig = plt.figure(figsize=(8, 8),constrained_layout=False) 
    gs = fig.add_gridspec(3, 3)
    gs.update(left=0.05, right=0.98, bottom = 0.12, top = 0.92, wspace=0.1, hspace=0.2)
    axs = []
    for ii in range(9):
        xx = ii%3
        yy = ii//3
        pt_itd = str(itd[yy])
        env_itd = str(itd[xx])
        # rate = str(pps[yy])
        # dur = str(durs[xx])
        ax = fig.add_subplot(gs[yy, xx])
        # ax.set_xlim(-5, 105)
        # ax.set_ylim(-5, 105)
        ax.set_title('ptITD'+pt_itd+'_envITD'+env_itd)
        axs.append(ax)
    return fig, axs

def plot3X3plot(swps, sub_conditions, chanNum, itd = [-0.1, 0, 0.1], allRepeats = False, avgRepeats = False,):
    sampleRate = swps[1].sampleRate
    duration = sub_conditions[0][1]
    clickRate = sub_conditions[1][1]
    t = np.arange(0, duration+0.05, 1/sampleRate)
    fig, axs = prepare3X3Plot()
    for ii in range(9):
        
        xx = ii%3
        yy = ii//3
        pt_itd = itd[yy]
        env_itd = itd[xx]
        ax = axs[ii]
        
        conditions=[('duration (s)',duration), ('clickRate (Hz)',clickRate), ('ITD (ms)',pt_itd), ('env ITD (ms)',env_itd)]
        swpIdx=findSweepsFor(conditions.copy(), stimPar)
        signal = poolChannel(swps[swpIdx], chanNum)
        if allRepeats == True:
            signal = (signal-signal[0, :])[:len(t), :]
        if avgRepeats == True:
            signal = signal.mean(axis = 1)[:len(t)]
        ax.plot(t, signal)
        ax.axvline(x = duration, color = 'k', linestyle='--')
    return fig

def prepare2X3Plot(durs = [0.01, 0.05, 0.2], pps = [900, 4500]):   
    fig = plt.figure(figsize=(12, 8),constrained_layout=False) 
    gs = fig.add_gridspec(2, 3)
    gs.update(left=0.05, right=0.98, bottom = 0.12, top = 0.92, wspace=0.1, hspace=0.15)
    axs = []
    for ii in range(6):
        xx = ii%3
        yy = ii//3
        rate = str(pps[yy])
        dur = str(durs[xx])
        ax = fig.add_subplot(gs[yy, xx])
        # ax.set_xlim(-5, 105)
        # ax.set_ylim(-5, 105)
        ax.set_title(rate+'pps_'+dur+'s')
        axs.append(ax)
    return fig, axs            
#%%
'''
VE distribution plot
'''

def plotScatter(xy, header, ax, text, color):
    # header = prepareHeader(sub_recordingSites, ii)
    # xy = prepareScatter(sub_recordingSites, header, jj)
    x = xy[header[2]]
    y = xy[header[1]]
    channel = xy[header[0]]
    for i in range(x.shape[0]):
        ax.text(x.iloc[i]*100, y.iloc[i]*100, text+'_ch'+str(channel.iloc[i]), alpha = 0.4, ha = 'center', rotation = 45) 
    ax.scatter(x*100, y*100, color = color, facecolors='none')
    ax.set_xlim(-5, 100)
    ax.set_ylim(-5, 100)

def Header(sub_recordingSites, ii, durs = [0.01, 0.05, 0.2], pps = [900, 4500]):
    xx = ii%3
    yy = ii//3
    rate = str(pps[yy])
    dur = str(durs[xx])
    columns = sub_recordingSites.columns
    VEheader,  significanceHeader = prepareHeader(columns, rate, dur)
    return VEheader, significanceHeader

def prepareHeader(columns, fac1, fac2):
    VEheader = [value for value in columns if 'VE' in value and fac1 in value and fac2 in value and '_significance' not in value and 'pValue' not in value or 'Channel' in value]
    significanceHeader = [value for value in columns if 'VE' in value and fac1 in value and fac2 in value and '_significance' in value]
    return VEheader, significanceHeader
    
def plot2X3Scatter(sub_recordingSites, fig, axs, text, durs = [0.01, 0.05, 0.2], pps = [900, 4500], colors=['tab:blue', 'tab:red', 'purple', 'grey']):
    # fig, gs_scatter = prepare2X3Plot()
    Label = [[1, 0], [0, 1], [1, 1], [0, 0]]
    for ii in range(6):
        ax = axs[ii]
        VEheader, significanceHeader = Header(sub_recordingSites, ii)
        for jj in range(4):
            df =  sub_recordingSites[(sub_recordingSites[significanceHeader[0]]==Label[jj][0]) & (sub_recordingSites[significanceHeader[1]]==Label[jj][1])]
            xy = df[VEheader]            
            plotScatter(xy, VEheader, ax, text, color = colors[jj])
        
    return fig
#%% 
'''
step one, generate inventory of data files for this study
The code below generated a CSV file that serves as inventory of all datafiles. 
The inventory CSV file is edited by hand to throw out bad penetrations
Step one achieved, now not needed, therefore commented out
'''
# shiyiAnims=[ # note that some animals are commented out here because their data are no good
#     #'2021_12_09', 
#     '2021_12_16', 
#     '2022_01_11', 
#     '2022_01_24',
#     '2022_01_25', 
#     '2022_01_27', 
#     #'2022_02_11', 
#     # '2022_02_16',
#     '2022_02_18', 
#     '2022_02_21', 
#     # '2022_02_24', 
#     '2022_02_25', 
#     '2022_03_02', 
#     # '2022_06_02', 
#     '2022_08_22', 
#     '2022_08_24',
#     '2023_02_16',
#     '2023_02_21',
#     '2023_02_23',
#     '2023_03_03']

# NHanims=[ 
#     '2021_12_16', 
#     '2022_01_11', 
#     '2022_02_16',
#     '2022_02_18', 
#     '2022_02_21', 
#     '2022_03_02', 
#     '2022_08_22']

# ITD_plus_anims=['CI2225_ITD_ENV_vs_PT_CI_1',
#   'CI2226_ITD_ENV_vs_PT_CI_1',
#   'CI2302_ITD_ENV_vs_PT_CI_1',
#   'CI2303_ITD_ENV_vs_PT_CI_1',
#   'CI2306_ITD_ENV_vs_PT_CI_1']

# ITD_BurstJitter_anims=['CI2224_ITD_ENV_vs_PT_CI_1',
#   'CI2228_ITD_ENV_vs_PT_CI_1']


# ITD_minus_anims=['CI2220_ITD-_ENV_vs_PT_CI_1',
# 'CI2217_ITD-_ENV_vs_PT_CI_1',
# 'CI2304_ITD-_ENV_vs_PT_CI_1',
# 'CI2229_ITD-_ENV_vs_PT_CI_1',]


# allAnims=shiyiAnims+ITD_plus_anims+ITD_BurstJitter_anims+ITD_minus_anims
# # all recordings for shiyiAnims are from Left IC unless there is RIC in the filename.
# # all recordings from Tim (not in shiyiAnims are Right IC)
# dirList=[]
# fnameList=[]
# condList=[]
# rightIC=[]
# for anim in allAnims:
#     ephysFiles=glob.glob(dataRoot+'/'+anim+'/*.ephys')
#     for efile in ephysFiles:
#         fnameList.append(os.path.basename(efile))
#         dirList.append(anim)
#         if anim in NHanims:
#             condList.append('NH')
#             if efile.find('RIC')>0:
#                 isRight=True
#             else:
#                 isRight=False
#         elif anim in ITD_plus_anims:
#             condList.append('ITD+')
#             isRight=True
#         elif anim in ITD_BurstJitter_anims:
#             condList.append('BurstJitter')
#             isRight=True
#         elif anim in ITD_minus_anims:
#             condList.append('ITD-')
#             isRight=True
#         else:
#             condList.append('ND')
#             if efile.find('RIC')>0:
#                 isRight=True
#             else:
#                 isRight=False
#         rightIC.append(isRight)

# #%
# dataset=pd.DataFrame({'fileName':fnameList,'animDir':dirList,
#                       'condition':condList,'rightIC':rightIC})

# dataset.to_csv(analysisRoot+'/datafileInventory.csv', index=False)
#%%
'''
Go through all recording data files listed in the inventory
and pre-process by doing artefact rejection and AMUA calculation.
Save pre-processed data to _CleanSig.ephys files
Save AMUA results to _AMUA.ephys
'''
# dataset=pd.read_csv(analysisRoot+'/datafileInventory.csv')

# for row in np.arange(len(dataset)): # set through file list
#     #load ephys data
#     if row > 27 and row < 59:

#         swps, stimPar = readEphysFileFromRow(dataset, row, loadOrig=True)
        
#         if isinstance(swps, list):
#             continue
        
#         # apply artifact rejection
#         #for Freiburg data: bring sweeps to one legth
#         # swps, stimPar = checklongswps(swps, stimPar)  # This funciton does not work...
#         swpClean, stimPar = ep.CIartefactRejectByAveraging(swps, stimPar) 
#         # detect and remove jump from signal after cleaning swps
#         # if row >= 59:
#         # swpClean = remove_jump(swpClean) 
#         writeEphysFileFromRow(dataset, row, swpClean, stimPar, writeClean=True)
        
#         # apply AMUA filter
#         # save AMUA results to AMUA.ephys
#         swpAMUA = [] 
#         for idx, swp in enumerate(swpClean):   
#             # apply AMUA filter and save AMUA results to AMUA.ephys
                   
#             # first swp possible shorter than AMUA padLen bring up error
#             # fake first AMUA swp
#             if idx == 0:
#                 fakeswp = ep.MCsweep(2000)
#                 fakeswp.signal = swpClean[0].signal
#                 swpAMUA.append(fakeswp)
#             else:
#                 swpAMUA.append(calcAMUA(swp, 2000))
#         writeEphysFileFromRow(dataset, row, swpAMUA, stimPar, writeAMUA=True)

#%% debug Feriburg data 
# leftoverFile = pd.read_csv(analysisRoot+'/LeftoverArtefact.csv')
# filename = 'CI2302_9_ITD_ENV_vs_PT_CI_1.ephys'
# swpsClean, stimPar=ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-6]+'_Clean')
# t = leftoverFile[filename].dropna()
# expIdx = int(t[0])

# fig, axs = prepare4X8Plot()
# stim = stimPar.iloc[expIdx]
# duration = stim['duration (s)']
# samplerate = swps[expIdx].sampleRate
# t = np.arange(0, duration+0.05, 1/samplerate)
# signal = swps[expIdx].signal
# for cc in range(32):
#     ax = axs[cc]
#     ax.plot(t, signal[:len(t), cc])
#%%
'''
Code to check if residual artefact is in the cleaned data: 
Give the function the claned sweeps and stim pars and it gives you a an array with the length of the sweeps. 
if the values in the array are 1 then there is residual artefact in the sweep
'''
def Certain (swps, stimPar):
    '''
    Parameters
    ----------
    swp : tAKE IN ONE Clean SWEEP- MCephys object.

    Returns: all the sweeps that are not good --> so if artefact is left. 
    -------
    swp : TYPE
        DESCRIPTION.

    '''
    sampleRate = swps[1].sampleRate
    Certain = np.zeros((len(swps)-1, 32))
    durations = stimPar["duration (s)"].values
    
    for swp in range (len(Certain)):
        # abs difference to overnext sample during stimulation time 
        for channel in range(32):
            # print(channel)
            differences = np.abs(np.diff(swps[swp+1].signal[:int(durations[swp]*sampleRate), channel], n = 2)) 
            std_tail = np.std(swps[swp+1].signal[-100:, channel])
            for idx, diff in enumerate (differences):
                if diff > 20 * std_tail:  
                    # print('yes')
                    Certain[swp][channel] = int(1)
    
    
    # indices = np.argwhere(Certain == 1)
    # start_time = time.time()
    Check_idx = [] #List of all channels and the sweeps with leftover artefacts of the respeccitve channel,  len = 32 and len for each idx depends on no. of sweeps with artefact
    # temp = np.sum(Certain, axis = 1)
    for chan in range(32):
        Channelwise = []
        for i, val in enumerate (Certain[:,chan]):
            if val > 0:
                Channelwise.append(i+1) 
        Check_idx.append(Channelwise)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(elapsed_time)
    
    # for swp_art in range (0,10):
    #     print (swp_art)
    #     plt.figure()
    #     plt.plot((swpClean[Check_idx[swp_art]].signal[:,0]))
    
    conds = []  #List with all the conds for all the channels with leftover artefact, len = 32 and len for each idx depends on no. of sweeps with artefact
    for channel in range(32):
        Channelwise = []
        for swp in Check_idx[channel]:
            conds_swp = findConditionsForswp (swp, stimPar)
            Channelwise.append(conds_swp)
        conds.append(Channelwise)

    Check_idx_cond_all = []
    for i in range(32):
        Check_idx_cond = [f"Chan: {i} Sweep: {x} Conds: {y}" for x, y in zip(Check_idx[i], conds[i])]
        Check_idx_cond_all.append(Check_idx_cond)
        
    flattened_list = [item for sublist in Check_idx_cond_all for item in sublist]
    
    
    return flattened_list



dataset=pd.read_csv(analysisRoot+'/datafileInventory.csv')
anims = []
artefact_left = []
for row in range (len(dataset)):
    
    if dataset.Cleaned[row] == 0:
        continue
    else:
        filename = dataset.iloc[row]['fileName']
        anim = dataset.iloc[row]['animDir']
        swpsClean, stimPar=ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-6]+'_Clean')
        left_art = Certain(swpsClean, stimPar)
        anims.append(filename)
        artefact_left.append(left_art)


#Save as pandas file
max_len = max(len(item) for item in artefact_left)

for sublist in artefact_left:
    sublist.extend([None] * (max_len - len(sublist)))

df = pd.DataFrame({anim: artefact_left[i] for i, anim in enumerate(anims)})

df.to_csv(analysisRoot+'/'+'LeftoverArtefact.csv', index=False) 

#%% 
'''
Generate inventory of recoding sites for the study
The code below generated a CSV file that serves as inventory of recording sites from cleaned datafile. 
'''
# dataset=pd.read_csv(analysisRoot+'/datafileInventory.csv')
# penetrationNameList = []
# channelList = []
# Right_IC_List = []
# Conditions = []
# for row in np.arange(len(dataset)):
#     if dataset.Cleaned[row] == 0:
#         continue
#     else:
#         penetrationNameList = penetrationNameList+[dataset.fileName[row]]*32
#         channelList = channelList+np.arange(32).tolist()
#         Right_IC_List = Right_IC_List + [dataset.rightIC[row]]*32
#         Conditions = Conditions + [dataset.condition[row]]*32
# df = pd.DataFrame({'fileName': penetrationNameList,'Channel': channelList, 'rightIC': Right_IC_List, 'condition': Conditions})
# df.to_csv(analysisRoot+'/'+'recordingsiteInventory.csv', index=False) 
#%% 
'''
run Wilcoxon-test over 'datafileInventory.csv'
save results back to 'unitInventory.csv'
'''
# dataset=pd.read_csv(analysisRoot+'/datafileInventory.csv')
# conditions=[('duration (s)',0.01), ('clickRate (Hz)',900), ('ITD (ms)',0.1), ('env ITD (ms)',-0.1)]
# wilcoxonResultsList = []
# selectionResultsList = []
# for row in np.arange(len(dataset)):
#     swpsAMUA, stimPar = readEphysFileFromRow(dataset, row, loadAMUA = True)
#     if isinstance(swpsAMUA, list):
#         continue
#     else:
#         results, selection = wilcoxonTestfor(swpsAMUA, stimPar, conditions)
#         wilcoxonResultsList = wilcoxonResultsList+results
#         selectionResultsList = selectionResultsList+selection
# df = pd.read_csv(analysisRoot+'/recordingsiteInventory_leftover.csv')
# df['wilcoxon-pValue'] = wilcoxonResultsList
# df['Responsive'] = selectionResultsList
# df.to_csv(analysisRoot+'/'+'recordingsiteInventory_leftover.csv', index=False)   

#%%
#Function to check which sweeps have left over artefact:

def LeftOverArtefact (filename, channel, rejected_art, stimPar):
    leftoverart = rejected_art[filename].dropna()
    if len(leftoverart) == 0:
        remove_keys = np.empty(0, dtype=object)
        return remove_keys
    if len(leftoverart) > 0:
        leftoverart_chan = leftoverart[leftoverart.str.contains(f"Chan: {channel}", na=False)]
        
        sweeps_art = []
        for entry in leftoverart_chan:
            sweep_art = int(re.search(r"Sweep: (\d+)", entry).group(1))
            sweeps_art.append(sweep_art)
        
        
        Artefact_check = []     #Holds combinations with too much artefact
        nup = ep.nonUniqueParams(stimPar)
        p = ep.permutations(nup, reverse = True)
        for ind, stim in enumerate(p):
            idxs_prev = ep.sweepIndecesFor(stimPar, list(nup.keys()), stim)
         
         #check if there is residual artefact:
            idxs = np.array([item for item in idxs_prev if item+1 not in sweeps_art])
         #Check if left over sweeps are at least 2/3 of the trials presented. 
            if len(idxs) < 0.75 * len(idxs_prev):
                # print('Too much leftover aretefact in combination: ', stim)
                Artefact_check.append(stim)
    
     #After the loop below: holds all the values fo keys that need to be removed:
        remove_keys = np.empty(len(Artefact_check), dtype=object)
        for i in range(len(Artefact_check)):
            remove_keys[i] = ('Dur'+str(Artefact_check[i][0])+'_Rate'+str(Artefact_check[i][1]))   
        remove_keys = np.unique(remove_keys)
        # print('Too much leftover aretefact in combination: ', remove_keys)
        return remove_keys


def remove_single_ARTsweeps(idxs_prev, filename, channel, rejected_art):
    leftoverart = rejected_art[filename].dropna()
    leftoverart_chan = leftoverart[leftoverart.str.contains(f"Chan: {channel}", na=False)]
    
    sweeps_art = []
    for entry in leftoverart_chan:
        sweep_art = int(re.search(r"Sweep: (\d+)", entry).group(1))
        sweeps_art.append(sweep_art)
        
    idxs = [item for item in idxs_prev if item+1 not in sweeps_art]
    idxs = np.array(idxs, dtype=int)
    
    return idxs


#%%
'''
variance explained calculation
'''
# read in datafileInventory.csv and recordingsiteInventory.csv
dataFiles = pd.read_csv(analysisRoot+'/datafileInventory.csv')
recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventory.csv')
rejected_art = pd.read_csv(analysisRoot+'/LeftoverArtefact.csv')
for row in range(len(dataFiles)):
    # row = 61
    if dataFiles.iloc[row]['Cleaned'] == 0:
        continue
    swpAMUA, stimPar = readEphysFileFromRow(dataFiles, row, loadAMUA = True)   
    # from datafilename find corresponding recording sites in recordingsiteInventory.csv
    sub_recordingSites = findRecordingSite(dataFiles, row)   
    
    #Check the sweeps with leftover artefact:
    filename = dataFiles.iloc[row]['fileName']
    nup = ep.nonUniqueParams(stimPar)
    p = ep.permutations(nup, reverse = True)
    
    # loop over sub_recordingSites, if not, fill blank
    for channel in sub_recordingSites.Channel:
        
        print ('Now analysing:', filename, 'channel:', channel)        
            
        #Remove keys if there are residual artefacts in more than 1/4 of the sweeps. 
        # Holds all the keys that need to be removed for this channel        
        remove_keys = LeftOverArtefact (filename, channel, rejected_art, stimPar)
        idx = sub_recordingSites.index[channel]
        for ii, stimulus in enumerate(p):
            
            keyName = 'Dur'+str(p[ii, 0])+'_Rate'+str(p[ii, 1])
            if keyName in remove_keys:
                    continue
            
            idxs_prev = ep.sweepIndecesFor(stimPar, list(nup.keys()), stimulus)            
            #check if there is residual artefact and remove single trials if less more than 1/4 still good:                         
            idxs = remove_single_ARTsweeps (idxs_prev, filename, channel, rejected_art)

            signal = poolChannel(swpAMUA[idxs+1], channel)[:int(stimulus[0]*swpAMUA[0].sampleRate), :].mean(axis = 0)
             
            if ii % 9 == 0:
                df = pd.DataFrame({'PT_ITD': stimulus[2].repeat(len(signal)), 'ENV_ITD': stimulus[3].repeat(len(signal)), 'AMUA': signal})
            else:
                df = pd.concat([df,pd.DataFrame({'PT_ITD': stimulus[2].repeat(len(signal)), 'ENV_ITD': stimulus[3].repeat(len(signal)), 'AMUA': signal})], ignore_index=True)                 
            if ii % 9 == 8:
                # print(df)
                ss_pt, ss_env = calcVarianceExplained(df) 
                ss_pt_p, ss_env_p = permutationTestfor(df, 1000)
                
                recordingSites = writeValueinto(df, idx, ss_pt, ss_env, ss_pt_p, ss_env_p, keyName)

# recordingSites.to_csv(analysisRoot+'/'+'recordingsiteInventory.csv', index=False)
#%%
'''
plot the left over sweeps that are used for VE calculation since we check for leftover artefacts:
'''  
def plot3X3plot_Sanity(swps, sub_conditions, filename, rejected_art, chanNum, itd = [-0.1, 0, 0.1], allRepeats = False, avgRepeats = False,):
    sampleRate = swps[1].sampleRate
    duration = sub_conditions[0][1]
    clickRate = sub_conditions[1][1]
    t = np.arange(0, duration+0.05, 1/sampleRate)
    fig, axs = prepare3X3Plot()
    for ii in range(9):
        
        xx = ii%3
        yy = ii//3
        pt_itd = itd[yy]
        env_itd = itd[xx]
        ax = axs[ii]
        
        conditions=[('duration (s)',duration), ('clickRate (Hz)',clickRate), ('ITD (ms)',pt_itd), ('env ITD (ms)',env_itd)]
        idxs_prev=findSweepsFor(conditions.copy(), stimPar)
        leftoverart = rejected_art[filename].dropna()
        leftoverart_chan = leftoverart[leftoverart.str.contains(f"Chan: {chanNum}", na=False)]
        sweeps_art = []
        for entry in leftoverart_chan:
            sweep_art = int(re.search(r"Sweep: (\d+)", entry).group(1))
            sweeps_art.append(sweep_art)
        idxs = [item for item in idxs_prev if item not in sweeps_art]
        swpIdx = np.array(idxs, dtype=int)
        print(swpIdx)
        # check if there is residual artefact and remove single trials if less more than 1/4 still good:                         
        # swpIdx = remove_single_ARTsweeps (idxs_prev, filename, channel, rejected_art)
        
        signal = poolChannel(swps[swpIdx], chanNum)
        if allRepeats == True:
            signal = (signal-signal[0, :])[:len(t), :]
        if avgRepeats == True:
            signal = signal.mean(axis = 1)[:len(t)]
        ax.plot(t, signal)
        ax.axvline(x = duration, color = 'k', linestyle='--')
    return fig





def plot32Channel_Sanity(swps, conditions, allRepeats = False, avgRepeats = False, ncols = 8, nrows = 4):
    sampleRate = swps[1].sampleRate
    Duration = conditions[0][1]
    t = np.arange(0, Duration+0.05, 1/sampleRate)
    nchans = swps[1].signal.shape[1]
    swpIdx=findSweepsFor(conditions.copy(), stimPar)
    swpPlot = swps[swpIdx]
    
    fig, axs = prepare4X8Plot()
    for cc in range(nchans):
        ax = axs[cc]        
        signal = poolChannel(swpPlot, cc)
        if avgRepeats:
            signal = signal[:len(t), :].mean(axis = 1)
            ax.plot(t, signal)
        if allRepeats:
            signal = (signal-signal[0, :])[:len(t)]
            ax.plot(t, signal)

        ax.axvline(x = Duration, color = 'k', linestyle='--')
        if not np.mod(cc,ncols) == 0:
            plt.yticks([])
        if not cc>=nchans-ncols:
            plt.xticks([])
    return fig



# dataset=pd.read_csv(analysisRoot+'/datafileInventory.csv')
row = 61
filename = dataset.iloc[row]['fileName']
anim = dataset.iloc[row]['animDir']
recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventory.csv')
rejected_art = pd.read_csv(analysisRoot+'/LeftoverArtefact.csv')

swps, stimPar = ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-6])
swpsClean, stimPar=ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-6]+'_Clean')
swpsAMUA, stimPar=ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-6]+'_AMUA')

check_chan = 5
check_Clickrate = 4500
check_dur = 0.01

# # # pool out condition index
conditions=[('duration (s)',check_dur), ('clickRate (Hz)',check_Clickrate), ('ITD (ms)',0.1), ('env ITD (ms)',0)]

# # # plot 32ch clean sig(all repeat)
# fig = plot32Channel_Sanity(swpsClean, conditions, allRepeats = True)
# fig.suptitle(filename[:-6]+'_clean'+str(conditions))
# # plt.savefig(figureRoot+'/'+dataset.fileName[row][:-6]+'_CleanSig') 
# # plt.close()

# # #plot 32ch AMUA(average)
# fig = plot32Channel_Sanity(swpsAMUA, conditions, avgRepeats = True)
# fig.suptitle(filename[:-6]+'_AMUA'+str(conditions))
# # plt.savefig(figureRoot+'/'+dataset.fileName[row][:-6]+'_AMUA')
# # plt.close()

# #plot 1ch clean sig, 3X3 itd 
conditions = [('duration (s)',check_dur), ('clickRate (Hz)',check_Clickrate)]
chanNum = check_chan

fig = plot3X3plot_Sanity(swpsClean, conditions, filename, rejected_art, chanNum, allRepeats=True)
      
#%%
'''
box plots
'''
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
dataFiles = pd.read_csv(analysisRoot+'/datafileInventory.csv')
recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventory.csv')
corhorts = dataFiles.condition.unique()
animDir = dataFiles.animDir.unique()
#Get the headers
headers = recordingSites.columns
Data_header = headers[6:]
significance_headers = [col for col in Data_header if "significance" in col]
VE_headers = [col for col in Data_header if "significance" not in col and "pValue" not in col]
pValue_headers = [col for col in Data_header if "pValue" in col]

#jusit a quick and dirty way to get the headers for a plot... 
Headers_Plot = []
#This is to get the VE letters --> Jan wated that for the plot
VE_kind = [header.split('_', 2)[-1] for header in VE_headers]
Dur_kind = [header.split('_', 1)[0] for header in VE_headers]
pps_kind = [header.split('_')[1] for header in VE_headers]


for i in (range(len(VE_headers))):
    # print (head)
    
    Headers_Plot.append(VE_kind[i] + ' ' + 'Dur:' + Dur_kind[i][3:] + ' pps:' + pps_kind[i][4:])

Anims = []
for anim in animDir:
    filenames = dataFiles[dataFiles['animDir']==anim]['fileName']
    sub_recordingSites = []
    for filename in filenames:
        sub_recordingSites.append(recordingSites[recordingSites['fileName']==filename])
    sub_recordingSites = pd.concat(sub_recordingSites, ignore_index=True)
    sub_recordingSites = sub_recordingSites[sub_recordingSites['Responsive'] == 1]
    Anims.append(sub_recordingSites)

#For a cohort:
for corhNum, corhort in enumerate(corhorts):
# if corhNum == 2:
    plt.figure()
    all_units = []
    All_data = []
    No_Animals = 0
    # All_labels = []
    for idx, anim in enumerate (Anims): 
        if idx == 22:
            continue
        # print (Anims[idx].condition.iloc[0])
        if Anims[idx].condition.iloc[0] == corhort:
            No_Animals += 1
            # print(corhort)
                 ###################################Check here how ,many UNITS!
            Units = len(anim)
            all_units.append(Units)
            All_data.append(anim[VE_headers]*100)
            # All_labels.append(Headers_Plot)
    
    Combined_Data = pd.concat(All_data)
    Combined_Data = [Combined_Data[col].dropna() for col in Combined_Data.columns]
    
    # Combined_Data = Combined_Data.dropna(axis=1, how='all')
    # Combined_Data = Combined_Data.fillna(1)
    combined_Units = np.sum(all_units)
    
    plt.boxplot(Combined_Data, labels = Headers_Plot)
    
    #For plotting the residual units as blue text below:
    for i, val in enumerate (Combined_Data):
        n = len(val)
        plt.text (i + 1, -3, f"Units : {n}", ha='center', fontsize=10, color='blue')
        
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.axvline(x=2.5, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=4.5, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=6.5, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=8.5, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=10.5, color='black', linestyle='--', linewidth=1)
            
    title = f"{corhort}, All units: {combined_Units} , No. animals: {No_Animals}"
    plt.title(title)
        
    plt.ylabel ('VE Values')
    plt.show()



#for single anims:
# for idx, anim in enumerate (Anims):
#     # print (anim)
    
#     # subject_data = recordingSites[recordingSites['fileName']].str.contains(anim).any()
        
#     # idx = 26
#     # anim = Anims[idx]
    
#     Units = len(anim)
#     # recordingSites = recordingSites.iloc[:32]
    
#     plt.figure()
    
#     plt.boxplot(anim[VE_headers]*100, labels = Headers_Plot)
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.axvline(x=2.5, color='black', linestyle='--', linewidth=1)
#     plt.axvline(x=4.5, color='black', linestyle='--', linewidth=1)
#     plt.axvline(x=6.5, color='black', linestyle='--', linewidth=1)
#     plt.axvline(x=8.5, color='black', linestyle='--', linewidth=1)
#     plt.axvline(x=10.5, color='black', linestyle='--', linewidth=1)
    
#     title = f"{animDir[idx]}, Units: {Units}, Condition: {anim['condition'].iloc[0]}"
#     plt.title(title)
    
#     plt.ylabel ('VE Values')
#     plt.show()


#%%
'''
VE distribution plot
'''
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
dataFiles = pd.read_csv(analysisRoot+'/datafileInventory.csv')
recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventory.csv')

corhorts = dataFiles.condition.unique()

for corhNum, corhort in enumerate(corhorts):
    #Only ITD+ anims for sanity check:
    if corhNum == 2: 
        fig, axs = prepare2X3Plot()
        anims = dataFiles[dataFiles['condition'] == corhort]['animDir'].unique()
        print (anims)
        unitNum = 0
        for animNum, anim in enumerate(anims):
            filenames = dataFiles[(dataFiles['animDir'] == anim) & (dataFiles['Cleaned'] == 1)]['fileName']
            for penNum, filename in enumerate(filenames):
                sub_recordingSites = recordingSites[(recordingSites['fileName'] == filename) & (recordingSites['Responsive'] == 1)]
                unitNum+=len(sub_recordingSites)
                scatterText = 'anim'+str(animNum)+'pen'+str(penNum)
                # print('animNum:', animNum, 'penNum:', penNum, 'filename:', filename)
                fig = plot2X3Scatter(sub_recordingSites, fig, axs, scatterText)
            fig.text(0, 0.5, 'Variance explained by PT_ITD (%)', va='center', rotation=-90, fontsize = 14)
            fig.text(0.38, 0.053, 'Variance explained by ENV_ITD (%)', va='center', fontsize = 14)
            fig.suptitle(corhort+'_ResponsiveUnit#'+str(unitNum), fontsize = 14)
            # plt.savefig('/ephys/2/0_ENVvsPT/Figure/Result_ScatterPlot/'+corhort)

#%%
'''
debug VE distribution
'''
corhort = 'ITD+'
animNum = 3
penNum = 6
chanNum = 16
duration = 0.01
clickRate = 900

anim = dataFiles[(dataFiles['condition'] == corhort)]['animDir'].unique()[animNum]
filename = dataFiles[(dataFiles['animDir'] == anim) & (dataFiles['Cleaned'] == 1)].iloc[penNum]['fileName']
swpsClean, stimPar=ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-6]+'_Clean')
swpsAMUA, stimPar=ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-6]+'_AMUA')

#%% plot clean data
sampleRate = swpsClean[1].sampleRate
t = np.arange(0, duration+0.05, 1/sampleRate)
fig, axs = prepare3X3Plot()

itd = [-0.1, 0, 0.1]
for ii in range(9):    
    xx = ii%3
    yy = ii//3
    pt_itd = itd[yy]
    env_itd = itd[xx]
    ax = axs[ii]
    print(ax.title, pt_itd, env_itd)
    conditions=[('duration (s)',duration), ('clickRate (Hz)',clickRate), ('ITD (ms)',pt_itd), ('env ITD (ms)',env_itd)]
    swpIdx=findSweepsFor(conditions.copy(), stimPar)
    signal = poolChannel(swpsClean[swpIdx], chanNum)
    signal = (signal-signal[0, :])[:len(t)]
    # signal = (signal-signal[0, :])[:len(t)].mean(axis = 1)
    ax.plot(t, signal)
    ax.axvline(x = duration, color = 'k', linestyle='--')
















