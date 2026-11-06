#!/usr/bin/env python3  
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 16:24:15 2024

@author: shiyi, Tim
"""
import config  # data/figure folders, see config.py
from tqdm import tqdm
# import glob
# import os
import pandas as pd 
import matplotlib.pyplot as plt
from sys import path
path.append(config.EPHYS_LIB+'/')
import RZ2ephys as ep
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
from io import IOBase
import struct
from scipy.signal import butter, lfilter, lfilter_zi, filtfilt, decimate
from scipy.fft import fft, fftfreq, ifft
import sys
import time
import copy
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
# import gc
import psyPhysConfig as config
from scipy.signal import detrend
import os
# import statistics
# import re
# from scipy.stats import zscore
# import tkinter as tk
# from tkinter import filedialog
# import seaborn as sns

'''
################################################################################################################################
This code is a mastercode that reads in all ephys data and cleans artefacts (Step 1), as well as calculates AMUA (Step 2)
further analysis is done for the respective config itself. 
--> Functions and idea taken from AAA_masterAnalysisScript.py
################################################################################################################################
'''

#%

#Read in stim Var by hand- not needed essentially but can be used:
# stimVarParams=['duration (s)', 'clickRate (Hz)', 'ITD (ms)', 'env ITD (ms)'] # specify the stim params that were permuted to generate different stimulus conditions
# stimVarParams=['ITD (ms)','ILD (dB)','clickRate (Hz)'] # specify the stim params that were permuted to generate different stimulus conditions

dataRoot=config.DATA_ROOT+'/'
# dataRoot='/ephys/2/CIproject/2_ITD_ILD_cueTrading/Data'
# analysisRoot=config.ANALYSIS_ROOT+'/Salt_Beaker_CSV_Files'
# analysisRoot=config.ANALYSIS_ROOT+'/Debugging_csv_files'
analysisRoot=config.ANALYSIS_ROOT+'/'

figureRoot = config.FIGURE_ROOT
stimVarParams = ['duration (s)', 'clickRate (Hz)', 'ITD (ms)', 'env ITD (ms)']
paramsDict = {'duration (s)': [0.01, 0.05, 0.2], 'clickRate (Hz)': [900.0, 4500.0], 'ITD (ms)': [-0.1, 0.0, 0.1], 'env ITD (ms)': [-0.1, 0.0, 0.1]}
#%
'''
write down the important conditions before starting the pipeline- so adjust these parameters and tell the program which data is going to be analyzed. Call this variable: "Data_conds"
makes life maybe easier. 
# Cue Trading: [('clickRate (Hz)',1, 100, 900]),[ ('ITD (ms)',[-120, -80, -40, 0, 40, 80, 120]), ('ILD',[-4, -1, 0, 1, 4])]
Env Vs Pt:  [('duration (s)',[0.01, 0.05, 0.2]), ('clickRate (Hz)',[900, 4500]), ('ITD (ms)',[-0.1, 0, 0.1]), ('env ITD (ms)',[-0.1, 0, 0.1])]
Weighting: ?

Code below loads one example csv file such that you know automatically the analysisRoot and dataRoot and figureRoot:
'''
# dataRoot= None
# analysisRoot= None
# figureRoot = None

# if dataRoot == None: 
#     root = tk.Tk()
#     root.withdraw()  # Hide the root window
#     dataRoot = filedialog.askdirectory(title="Select a data root", initialdir = '/ephys/2')
# if analysisRoot == None:
#     root = tk.Tk()
#     root.withdraw()  # Hide the root window
#     analysisRoot = filedialog.askdirectory(title="Select a analysis root", initialdir = dataRoot)
# if figureRoot == None:
#     root = tk.Tk()
#     root.withdraw()  # Hide the root window
#     figureRoot = filedialog.askdirectory(title="Select a figure root", initialdir = dataRoot)
# root = tk.Tk()
# root.withdraw()  # Hide the root window
# file_path = filedialog.askopenfilename(title="Select a CSV file", initialdir=dataRoot, filetypes=[("CSV files", "*.csv")])
# df = pd.read_csv(file_path)
# columnNames = df.columns.tolist()
# while True: 
#     stimVarParams = []
#     for column in columnNames:
#         keep = input(f"Is '{column}' variable to be analyzed? (y/n): ").strip().lower()
#         if keep == 'y':
#             stimVarParams.append(column)
#     happy = input(f"Variable to be analyzed is {stimVarParams}, are you happy?(y/n)")
#     if happy == 'y':
#         break

# paramsDict = {}

# # Store unique values for each selected parameter
# for param in stimVarParams:
#     uniqueVals = sorted(df[param].unique().tolist())
#     paramsDict[param] = uniqueVals

#%
'''
Read in & write in
'''
def readEphysFileFromRow(dataset, row, stimVarParams = stimVarParams, dataRoot = None, loadOrig = False, loadAMUA = False, loadClean = False):
    if loadOrig == True:
        filename = dataset.fileName[row]
    if loadAMUA == True:
        filename = dataset.fileName[row][:-6]+'_AMUA'
    if loadClean == True:
        filename = dataset.fileName[row][:-6]+'_Clean'
    if not any('Cleaned' in col for col in dataset.columns):# if no 'Cleaned' column in csv file, read in all row
        swps, stimPar = ep.readEphysFile(dataRoot+'/'+dataset.animDir[row]+'/'+filename)
        if loadOrig == True:
            swps, stimPar = checkSwpsSize(swps, stimPar, stimVarParams)
        onlyKeepFirst32channels(swps)
        checkIfLastSweepIsMissing(swps, stimPar)
        print(filename)
    if any('Cleaned' in col for col in dataset.columns):# if exist 'Cleaned' column in csv file, only read in row labed as '1' in cleaned
        if dataset.Cleaned[row] == 1:
            swps, stimPar=ep.readEphysFile(dataRoot+'/'+dataset.animDir[row]+'/'+filename)
            if loadOrig == True:
                swps, stim1Par = checkSwpsSize(swps, stimPar, stimVarParams)
            onlyKeepFirst32channels(swps)
            print(filename)
        if dataset.Cleaned[row] == 0:
            swps = []
            stimPar = []
    return swps, stimPar

def writeEphysFileFromRow(dataset, row, swps, stimPar, dataRoot = None, writeAMUA = False, writeClean = False):
    filename = dataset.fileName[row]
    if filename.endswith('.ephys'):
        filename = filename[:-6]
    if writeAMUA == True:
        filename = filename+'_AMUA'
    if writeClean == True:
        filename = filename+'_Clean'
        
    # Check and create directory if it doesn't exist
    directory = os.path.join(dataRoot, dataset.animDir[row])
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    ep.writeEphysFile(os.path.join(directory, filename), swps, stim = stimPar)

def writeCsvFileFromRow(dataset, row, writeWilcoxon = False):
    if writeWilcoxon == True:
        filename = dataset.fileName[row][:-6]+'.csv'
    dataset.to_csv(dataRoot+'/'+dataset.animDir[row]+'/'+filename, index=False)
    

#%
'''
some useful function
'''

def onlyKeepFirst32channels(swps):
    # some data files have 32 channel IC data plus 64 chan ctx data
    # here we only want to analyse the first 32 chan IC data
    if swps[1].signal.shape[1] != 32:
        for swp in swps:
            swp.signal = swp.signal[:, :32]

def checkIfLastSweepIsMissing(swps, stimPar):            
    if len(swps) == len(stimPar):
        # There should be one more sweep than stim params
        # Sometimes the last sweep was not recorded though
        # In that case, drop the last stimPar from the stimPar table
        stimPar = stimPar.iloc[:-1]
    # Finally, check that lengths are now as expected
    if len(swps) != len(stimPar)+1:
        raise Exception('len(swps) does not equal len(stimPar)+1. Something is wrong with this data file!')

def checkSwpsSize(swps, stimPar, stimVarParams):
    # for HK data, some set recorded by 64 channel electrode containing IC and AC, only take first 32ch(IC)
    # for FR data, some set last swp is missed or shorter than normal, replace by second last swp
    onlyKeepFirst32channels(swps)

        
    # Code by Tim to get rid of the fact that sometimes the one sweep was too short (2* stim duration in sec):
    # Can also adjust the duration sweeps has to have 
    Short_swps = [] #List of short swps, adjusted by the for loop below... 
    for idx in  range (len(stimPar)):
        if swps[idx+1].signal.shape[0] < 200 + (float(stimPar.iloc[idx]['duration (s)']))*swps[-1].sampleRate:
        # if swps[idx+1].signal.shape[0]/swps[-1].sampleRate < 2 * (float(stimPar.iloc[idx]['duration (s)'])):
            # If it is smaller than 200 samples after duration 
            Short_swps.append(idx)
            print('sweeps too short:', idx)
    
    for i, swp in enumerate (Short_swps):
        swps = np.delete(swps, swp+i+1)
    
    stimPar2 = stimPar.drop(Short_swps, axis = "index")
    
    return swps, stimPar2
    
    """ 
    This next bit is BAD! It duplicates sweeps! It doesn't drop sweeps deemed "too short". That falsifies data.
    """
    # for idx, Short_swp in enumerate (Short_swps):
    #     # print(Short_swp)
    #     Short_conds = findConditionsForswp(Short_swp+1, stimPar, stimVarParams)
    #     # print(Short_conds)
    #     Last_swpIdxs = findSweepsFor(Short_conds, stimPar)
    #     for i, Last_swpIdx in enumerate (Last_swpIdxs):
    #         if Last_swpIdx == Short_swp+1:
    #             try:
    #                 swps[Last_swpIdxs[i]] = swps[Last_swpIdxs[i - 1]]
    #             except: 
    #                 swps[Last_swpIdxs[i]] = swps[Last_swpIdxs[i + 1]]
    # return swps, stimPar  
    

def findLastCond(stimPar, stimVarParams):
    aStim=stimPar.iloc[len(stimPar)-1]
    Last_conds=[]
    for aCond in stimVarParams:
        Last_conds.append( (aCond, f'{aStim[aCond]}') )
    return Last_conds



def findSweepsFor(conditions, stimPar):
    # returns a list of indeces into for the sweeps that satisfy the conditions.
    # conditions should be a list,  of pairs (paramName, paramValue)
    # For now e only do equality, and logical AND, ie we look for sweeps
    # satisfying (paramName1==paramValue1) AND (paramName2==paramValue2) ...    
    if not type(conditions)==list:
        raise Exception("conditions must be a list of pairs, eg [('duration (s)',0.01),('clickRate (Hz)',900)]")
    acond=conditions.pop()
    select=stimPar[acond[0]]==float(acond[1])
    while len(conditions)>0:
        acond=conditions.pop()
        select=np.logical_and(select,stimPar[acond[0]]==float(acond[1]))
    swpIdx=np.where(select)[0]      
    return swpIdx + 1

def findConditionsForswp (swp, stimPar, stimVarParams):
    # print (swp)
    aStim=stimPar.iloc[swp-1]
    conds=[]
    for aCond in stimVarParams:
        conds.append( (aCond, f'{aStim[aCond]}') )
        # print(aCond)
    return conds        
    

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

# def findRecordingSite(dataFiles, row):
#     filename = dataFiles['fileName'][row]
#     sub_recordingSites = recordingSites[recordingSites['fileName']==filename]
#     sub_recordingSites = sub_recordingSites[sub_recordingSites['Responsive'] == 1]     
#     return sub_recordingSites

# def writeValueinto(df, idx, V1, V2, Vp1, Vp2, KeyName):
#     df = checkKeyNamein(df, keyName)
#     df.at[idx, keyName+'_VEpt'] = V1
#     df.at[idx, keyName+'_VEenv'] = V2                     
#     df.at[idx, '(pValue)'+keyName+'_VEpt'] = Vp1
#     df.at[idx, '(pValue)'+keyName+'_VEenv'] = Vp2
#     if V1>Vp1:
#         df.at[sub_recordingSites.index[channel], keyName+'_VEpt_significance'] = 1
#     else:
#         df.at[sub_recordingSites.index[channel], keyName+'_VEpt_significance'] = 0
#     if V2>Vp2:
#         df.at[sub_recordingSites.index[channel], keyName+'_VEenv_significance'] = 1
#     else:
#         df.at[sub_recordingSites.index[channel], keyName+'_VEenv_significance'] = 0
#     return df

# def checkKeyNamein(recordingSites, keyName):
#     if not any(keyName in col for col in recordingSites.columns):
#         recordingSites[keyName+'_VEpt'] = ''
#         recordingSites['(pValue)'+keyName+'_VEpt'] = ''
#         recordingSites[keyName+'_VEpt_significance'] = ''
#         recordingSites[keyName+'_VEenv'] = ''
#         recordingSites['(pValue)'+keyName+'_VEenv'] = ''
#         recordingSites[keyName+'_VEenv_significance'] = ''
#     return recordingSites

#%
"""  
AMUA Calculation
    
"""
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, csd, windows, welch, get_window, resample, lfilter


def AMUAFilterCoeffs(fs,lowpass=6000):
        nyq = 0.5*fs
        bBand,aBand = butter(2,(300/nyq, 6000/nyq),'bandpass')
        bLow,aLow = butter(2,(lowpass/nyq),'lowpass')
        bNotch, aNotch = iirnotch(50, 30, fs)
        return [[bBand, aBand], [bLow, aLow], [bNotch, aNotch]]
    
def calcAMUA(swp, coefs, Fs_downsample, padLen=500):
# def calcAMUA(fs, ori_signal, Fs_downsample, padLen=300):
    ## XXX Multiple inefficieincies here: 
        # why recompute teh filter coefficiients for every sweep?  --> Now coefs are computet in line 933 coefs=AMUAFilterCoeffs(fs)
        # why do a 50 Hz notch filter after a 300 Hz low-pass?    --> Now Noth is removed but we still have Bandpass prev. to Lowpass bc. I was not sure if I can start the AMUA directly with insig=np.abs(swp.signal)
        
        # Anyways if the Bandpass prev. to the Lowpass can be skipped by using insig=np.abs(swp.signal) feel free to do so! 
    fs = swp.sampleRate
    bpCoefs=coefs[0]
    lpCoefs=coefs[1]
    # NotchCoefs = coefs[2]
    # insig = filtfilt(NotchCoefs[0], NotchCoefs[1], swp.signal, axis=0, padlen=padLen)
    # insig = np.flip(insig)
    # insig=filtfilt(bpCoefs[0],bpCoefs[1], swp.signal, axis=0, padlen=padLen)
    insig=np.abs(swp.signal)
    insig=filtfilt(lpCoefs[0],lpCoefs[1],insig,axis=0, padlen=padLen)
    # insig = np.flip(insig)          
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


def remove_jump_RAW (swps, stimPar):
    '''
    Parameters
    ----------
    swp : tAKE in several sweeps as MCephys object.

    Returns: A cleaned version of the sweep in case jumps happen that are bigger than 10*std if they happen after the stimulus
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
            differences = np.abs(np.diff(swps[i].signal[int(durations[i-1]*sampleRate) + 10 :,cc], n = 5))
            threshold = np.mean(differences) + 10 * np.std(differences)  #adjust here the criterion to find the jumps, right now its 5times std...
            jump_indices = np.where(differences > threshold)[0] + 12 # get the indeces where jumps are 
            jump_indices = jump_indices + int(durations[i-1]*sampleRate)
            jump_indices = np.min(jump_indices)
            for jump_index in jump_indices:
                # print(idx)
                print("JUMP DETECTED in" + str(i))
                shift = swps[i].signal[jump_index -1 ,cc] - swps[i].signal[jump_index ,cc]
                swps[i].signal[jump_index:,cc] += shift
    return swps


#%
'''
Wilcoxon Test
'''


def wilcoxonTestfor(swps, stimPar, conditionToCheck):
    swpIdx=findSweepsFor(conditionToCheck.copy(), stimPar)
    alen = np.zeros([len(swpIdx)])
    for ss in range(len(swpIdx)): 
        alen[ss]=swps[swpIdx[ss]].signal.shape[0]
    minLen=int(np.min(alen)) -1
    swpTest = swps[swpIdx]
    sampleRate = swpTest[0].sampleRate
    nchans = swpTest[0].signal.shape[1]
    # duration = conditionToCheck[0][1]
    duration = stimPar.iloc[swpIdx[0]]['duration (s)']
    responseWinLen=int(duration*sampleRate)
    baselineWinLen=int(0.2*sampleRate)
    selectionArray = []
    resultsArray = []
    for cc in range(nchans):
        signal = poolChannel(swpTest, cc)
        responseWinAvg = signal[:responseWinLen, :].mean(axis = 0)
        # baselineWinAvg = signal[minLen-int(duration*sampleRate):minLen, :].mean(axis = 0) ################################## Adjusted Wilcoxon!!!####################################
        baselineWinAvg = signal[-baselineWinLen:, :].mean(axis = 0) 
        ################################## Adjusted Wilcoxon!!!####################################
        results = stats.wilcoxon(responseWinAvg, baselineWinAvg)
        resultsArray.append(results[1])
        if results[1] < 0.05:
            selectionArray.append(1)
        else:
            selectionArray.append(0)
    return resultsArray, selectionArray

#%
'''
variance explained ratio calculation     
'''   
def calcVarianceExplained(df):
    ss_t = ssTotalCalc(df)
    model = ols('AMUA ~ C(PT_ITD)+C(ENV_ITD)+C(PT_ITD):C(ENV_ITD)', data = df).fit()
    # model = ols('AMUA ~ C(PT_ITD)+C(ENV_ITD)+C(PT_ITD):C(ENV_ITD)', data = df).fit()
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
#%
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
#%
'''
plots
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

# def remove_50hz(data, fs, q=30):
#     """
#     Notch filter to remove 50 Hz noise.
    
#     data : input signal
#     fs   : sampling frequency (Hz)
#     q    : filter sharpness (higher = narrower notch)
#     """
#     w0 = 50 / (fs / 2)         # Normalize frequency
#     b, a = iirnotch(w0, q)     # Design filter
#     filtered = filtfilt(b, a, data)
#     return filtered
from scipy.fft import fft, fftfreq, ifft



def remove_50hz(signal, fs, width=1.0):
    """
    Remove 50 Hz by zeroing FFT bins.
    
    signal : 1D array
    fs     : sampling rate
    width  : width of the notch in Hz
    """
    N = len(signal)
    freqs = np.fft.rfftfreq(N, 1/fs)

    # FFT
    spectrum = np.fft.rfft(signal)

    # Identify bins near 50 Hz
    notch = (freqs >= 50 - width/2) & (freqs <= 50 + width/2)

    # Zero (or attenuate) them
    spectrum[notch] = 0

    # Reconstruct signal
    cleaned = np.fft.irfft(spectrum, n=N)
    return cleaned

#%
'''
Code to check if residual artefact is in the cleaned data: 
Give the function the claned sweeps and stim pars and it gives you a an array with the length of the sweeps. 
if the values in the array are 1 then there is residual artefact in the sweep
'''
# def Certain (swps, stimPar):
#     '''
#     Parameters
#     ----------
#     swp : tAKE IN ONE Clean SWEEP- MCephys object.

#     Returns: all the sweeps that are not good --> so if artefact is left. 
#     -------
#     swp : TYPE
#         DESCRIPTION.

#     '''
#     sampleRate = swps[1].sampleRate                     #get sample rate
#     Certain = np.zeros((len(swps)-1, 32))               #create array to write the certain values in --> shape swps x chans 
#     durations = stimPar["duration (s)"].values          #get durations 
    
#     for swp in tqdm(range (len(Certain))):
#         # abs difference to overnext sample during stimulation time 
#         for channel in range(32):
#             # print(channel)
#             differences = np.abs(np.diff(swps[swp+1].signal[:int(durations[swp]*sampleRate), channel], n = 2))  #Get the abs values of point n and n+2 in one seep for one channel
#             std_tail = np.std(swps[swp+1].signal[-100:, channel])                                               #calculate the std of the last 100 values of the corresponding sweep 
#             for idx, diff in enumerate (differences):
#                 if diff > 20 * std_tail:                       # if the abs difference is bigger than 20x std of the baseline we mark for this sweep a 1 in the Certain array
#                     # print('yes')
#                     Certain[swp][channel] = int(1)              # making the value 1 means leftover artefact 
    
    
#     # indices = np.argwhere(Certain == 1)
#     # start_time = time.time()
#     Check_idx = []                   #List of all channels and the sweeps with leftover artefacts of the respeccitve channel,  len = 32 and len for each idx depends on no. of sweeps with artefact
#     # temp = np.sum(Certain, axis = 1)
#     for chan in range(32):
#         Channelwise = []
#         for i, val in enumerate (Certain[:,chan]):
#             if val > 0:
#                 Channelwise.append(i+1) 
#         Check_idx.append(Channelwise)
#     # end_time = time.time()
#     # elapsed_time = end_time - start_time
#     # print(elapsed_time)
    
#     # for swp_art in range (0,10):
#     #     print (swp_art)
#     #     plt.figure()
#     #     plt.plot((swpClean[Check_idx[swp_art]].signal[:,0]))
    
#     conds = []                   #List with all the conds for all the channels with leftover artefact, len = 32 and len for each idx depends on no. of sweeps with artefact
#     for channel in range(32):
#         Channelwise = []
#         for swp in Check_idx[channel]:
#             conds_swp = findConditionsForswp (swp, stimPar)
#             Channelwise.append(conds_swp)
#         conds.append(Channelwise)

#     Check_idx_cond_all = []     # list of all chans, sweeps and conditions --> saved in csv file 
#     for i in range(32):
#         Check_idx_cond = [f"Chan: {i} Sweep: {x} Conds: {y}" for x, y in zip(Check_idx[i], conds[i])]
#         Check_idx_cond_all.append(Check_idx_cond)
        
#     flattened_list = [item for sublist in Check_idx_cond_all for item in sublist]
    
    
#     return flattened_list


#If you want to run the code above just uncomment the rest of this cell here: 

# dataset=pd.read_csv(analysisRoot+'/datafileInventory.csv')
# anims = []
# artefact_left = []
# print('Artefact rejection Quality control: ')
# for row in range (len(dataset)):
    
#     # if dataset.Cleaned[row] == 0:
#     #     continue
#     # else:
#         filename = dataset.iloc[row]['fileName']
#         print('Checking quality of artefact rejection in '+filename+"...")
#         anim = dataset.iloc[row]['animDir']
#         print('Loading data ...')
#         swpsClean, stimPar=ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-6]+'_Clean')
#         left_art = Certain(swpsClean, stimPar)
#         anims.append(filename)
#         artefact_left.append(left_art)
#         print()

# #Save as pandas file
# print('Saving QC results to '+analysisRoot+'/'+'LeftoverArtefact.csv')
# max_len = max(len(item) for item in artefact_left)

# for sublist in artefact_left:
#     sublist.extend([None] * (max_len - len(sublist)))

# df = pd.DataFrame({anim: artefact_left[i] for i, anim in enumerate(anims)})

# df.to_csv(analysisRoot+'/'+'LeftoverArtefact.csv', index=False) 
# print('... done')

#%% 
'''
# step one, generate inventory of data files for this study
# The code below generated a CSV file that serves as inventory of all datafiles. 
# The inventory CSV file is edited by hand to throw out bad penetrations
# Step one achieved, now not needed, therefore commented out

################################################################################################################################
uncomment this cell if csv file was created to make sure you don't accidentaly write it twice
################################################################################################################################

'''
# allAnims=[
# 'Neonatally_Deafened/20230216/',
# 'Neonatally_Deafened/20230221/',
# 'Neonatally_Deafened/20230223/',

# 'Neonatally_Deafened/20230303/',
# 'Neonatally_Deafened/20230307/',
# 'Neonatally_Deafened/20230309/',

# 'Neonatally_Deafened/20230315/',
# 'Neonatally_Deafened/20230320/',
# 'Neonatally_Deafened/20230321/',

# 'Normal_Hearing/20230509/',
# 'Normal_Hearing/20230511/',
# 'Normal_Hearing/20230515/',
# 'Normal_Hearing/20230525/',
# 'Normal_Hearing/20230530/',
# 'Normal_Hearing/20230606/',
# 'Normal_Hearing/20230608/']


# dirList=[]
# fnameList=[]
# cohortList=[]
# rightIC=[]

# for anim in allAnims:
#     ephysFiles=glob.glob(dataRoot+'/'+anim+'/*ITD_ILD_cue*.ephys')
#     for efile in ephysFiles:
#         fnameList.append(os.path.basename(efile))
#         dirList.append(anim)
#         # all recordings are from left IC except for 2 animals:
#         if anim in ['Normal_Hearing/20230509/','Normal_Hearing/20230525/']:
#             isRight=True
#         else:
#             isRight=False
#         rightIC.append(isRight)
#         # THere are two cohorts, NH and ND, as indicated by the path name
#         if anim.find('Normal') < 0:
#             cohortList.append('ND')
#         else:
#             cohortList.append('NH')
        
# dataset=pd.DataFrame({'fileName':fnameList,'animDir':dirList,
#                       'cohort':cohortList,'rightIC':rightIC})

# dataset.to_csv(analysisRoot+'/datafileInventory.csv', index=False)


#%%
'''
Go through all recording data files listed in the inventory
and pre-process by doing artefact rejection and AMUA calculation.
Save pre-processed data to _CleanSig.ephys files
Save AMUA results to _AMUA.ephys
'''
dataset=pd.read_csv(analysisRoot+'datafileInventory.csv')
# dataset=pd.read_csv(analysisRoot+'Salt_Beaker_CSV_Files/datafileInventory_Salt_Beaker_Agar.csv')
# dataset = dataset[(dataset.condition == 'NH')].reset_index(drop=True)
# dataset = dataset[dataset.fileName.str.contains('ICAC')]
for row in np.arange(len(dataset)): # set through file list
    #load ephys data
    # if row >= 111: # and row < 59: #  and row < 30: # for debugging, set range of rows here
    # if row >=28 and row < 59:
    if row == 32:
    # if row > 15:
    # if 'ICAC' in dataset.iloc[row].fileName:
    # if True: # otherwise run everything
        print('loading file '+ dataset.iloc[row].fileName+'...')
        # if 'ICAC' in dataset.iloc[row].fileName:
        #     print(f"Skipping {dataset.iloc[row].fileName} as it contains 'ICAC'")
        #     continue
        swps, stimPar = readEphysFileFromRow(dataset, row, stimVarParams, dataRoot = config.DATA_ROOT, loadOrig=True)
        
        if isinstance(swps, list): # comment by Jan: I don't know what this does. Seems suspect --> Answer from Tim: ths is to get rid of the data we considered "not clean" in the the datafileinventory by adding a 0 in the Cleaned column. 
            continue
        
        # remove 50Hz for saline data --> Not required anymore since we do Bandpass 300 to 4k
        # for ii, swp in enumerate(swps):
        #     if ii == 0:
        #         continue
        #     for cc in range(32): 
        #         swp.signal[:, cc] = remove_50hz(swp.signal[:, cc], fs=swp.sampleRate, width = 1)
        
        
        # apply artifact rejection now with a Bandpass 300 to 4k
        #for Freiburg data: bring sweeps to one legth
        # swps, stimPar = checklongswps(swps, stimPar)  # This funciton does not work...
        print('Computing artefact rejection ...')
        swpClean, stimPar = ep.CIartefactRejectByAveraging(swps, stimPar) 
        # detect and remove jump from signal after cleaning swps
        
        print('Writing cleaned data to disk ...') 
        # writeEphysFileFromRow(dataset, row, swpClean, stimPar, dataRoot = config.DATA_ROOT+'/', writeClean=True)
        writeEphysFileFromRow(dataset, row, swpClean, stimPar, dataRoot = config.FRESH_DIR, writeClean=True)
        
        #Write to the saved csv file with the ending _Clean the std of the last 100 recording points and the abs max difference value fom n to n+2
        # filename = dataset.fileName[row][:-5]+'_Clean.csv'
        filename = dataset.fileName[row]
        if filename.endswith('.ephys'):
            filename = filename[:-6]+'_Clean.csv'
        else:
            filename = filename+'_Clean.csv'
            #changed to new dataroot with freshly analyzed
        df = pd.read_csv(config.FRESH_DIR+'/'+dataset.animDir[row]+'/'+filename)
        # df = pd.read_csv(config.DATA_ROOT+'/'+dataset.animDir[row]+'/'+filename)

        for chan in range (32):
            chan = str(chan)
            # if 'std_chan' + chan not in df.columns:
            #     df['std_chan' + chan] = None  # Create the column with empty values and the header "std"
            # if 'Max_Abs_chan' + chan not in df.columns:
            #     df['Max_Abs_chan' + chan] = None  # Create the column with empty values and the header "Max_Abs_n_n2"
            if 'ABS_STD_DIV_chan' + chan not in df.columns:
                df['ABS_STD_DIV_chan' + chan] = None  # Create the column with empty values and the header "Teiler"
                df['DIV_baseline' + chan] = None  # Create the column with empty values and the header "Teiler"
            
        # stds = np.zeros((len(swps)-1, 32)) 
        sampleRate = swps[1].sampleRate                     #get sample rate
        Max_Abs_chan = np.zeros((len(swps)-1, 32))                   #create array to write abs/ std values of the duration time --> shape swps
        Max_Abs_chan_baseline = np.zeros((len(swps)-1, 32))            #create array to write abs/ std values of the baseline --> shape swps
        durations = stimPar["duration (s)"].values          #get durations 
        Rates = stimPar ["clickRate (Hz)"].values
        
        #Check to n n+2 with n n+2 from the tail, last 100 entrances
        # Tail_Max_Abs_chan = np.zeros((len(swps)-1, 32)) 
        ###
        
        for swp in tqdm(range (len(Max_Abs_chan))):
            # abs difference to overnext sample during stimulation time 
            for channel in range(32):
                # print(channel)
                
                if Rates[swp] == 1:
                    PPSamp = 1
                else:
                    PPSamp = int(np.round(sampleRate/Rates[swp])) # So many samples are one pulse
                
                
                # Calculate the triplets around the peak vals during stim time 
                # differences = np.abs(np.diff(swpClean[swp+1].signal[:int(durations[swp]*sampleRate), channel], n = 2))  #Get the abs values of point n and n+2 in one seep for one channel
                signal = swpClean[swp+1].signal[:int(durations[swp]*sampleRate), channel]
                differences = signal[4:] - signal[:-4]
                Max_Abs_Val_chan = np.max(differences)
                Max_Abs_Idx_chan = np.argmax(differences)+4 
                # print(Max_Abs_Val_chan, Max_Abs_Idx_chan)
                
                # plt.figure(); plt.plot(signal, '-o')
                # plt.plot(Max_Abs_Idx_chan, Max_Abs_Val_chan, 'ro')
                               

                if PPSamp == 1: # if pulserate = 1 there is no triplet
                    Max_Abs_chan[swp][channel] = Max_Abs_Val_chan
                    
                else:
                    #Decision tree if max value is directly at the beginning/ end of the recording:
                    if Max_Abs_Idx_chan-PPSamp < 0:
                        Triplets = np.abs(Max_Abs_Val_chan + differences[Max_Abs_Idx_chan+PPSamp] + differences[Max_Abs_Idx_chan+(2*PPSamp)]) #Create a triplet of the 2 peaks around our max peak 
                        Max_Abs_chan[swp][channel] = Triplets
                    
                    elif Max_Abs_Idx_chan+PPSamp >= len(differences):
                        Triplets = np.abs(Max_Abs_Val_chan + differences[Max_Abs_Idx_chan-PPSamp] + differences[Max_Abs_Idx_chan-(2*PPSamp)]) #Create a triplet of the 2 peaks around our max peak 
                        Max_Abs_chan[swp][channel] = Triplets
                    
                    else:
                        Triplets = np.abs(Max_Abs_Val_chan + differences[Max_Abs_Idx_chan+PPSamp] + differences[Max_Abs_Idx_chan-PPSamp]) #Create a triplet of the 2 peaks around our max peak 
                        Max_Abs_chan[swp][channel] = Triplets
                    
    #             #Following 6 lines are from tim to get rid of the "jumps" after cleaning due to minLen usage 
    #             #Calculate the same for the baseline as comparison  ########################################################
    #             # differences_baseline = np.abs(np.diff(swpClean[swp+1].signal[int(durations[swp]*sampleRate)+200:, channel], n = 2))  #Get the abs values of point n and n+2 in one seep for one channel--> +50 because 50 samples after stim ends starts the baseline-...
    #             Conds = findConditionsForswp (swp + 1, stimPar, stimVarParams = ['duration (s)', 'clickRate (Hz)', 'ITD (ms)', 'env ITD (ms)'])
    #             All_swps = findSweepsFor(Conds, stimPar)
    #             alen = np.zeros([len(All_swps)])
    #             for ss in range(len(All_swps)): 
    #                 alen[ss]=swps[All_swps[ss]].signal.shape[0]
    #             minLen=int(np.min(alen)) -1
    # ############################################################################################################################################
    
                # Shiyi has inbuild the minlen jump remove in the artifact rejection function
                
                # baseline = swpClean[swp+1].signal[int(durations[swp]*sampleRate)+200: minLen, channel]   # use max. minLen now here
                baseline = swpClean[swp+1].signal[int(durations[swp]*sampleRate)+200:, channel]
                differences_baseline = baseline[4:] - baseline[:-4]
                Max_Abs_Val_chan_baseline = np.max(differences_baseline)
                Max_Abs_Idx_chan_baseline = np.argmax(differences_baseline)+4
                # print(Max_Abs_Idx_chan_baseline, Max_Abs_Val_chan_baseline)
                
                # plt.figure(); plt.plot(baseline, '-o')
                # plt.plot(Max_Abs_Idx_chan_baseline, Max_Abs_Val_chan_baseline, 'ro')
                
                if PPSamp == 1: # if pulserate = 1 there is no triplet
                    Max_Abs_chan_baseline[swp][channel] = Max_Abs_Val_chan_baseline
                
                else:    
                    if Max_Abs_Idx_chan_baseline-PPSamp < 0:
                        Triplets_baseline = np.abs(Max_Abs_Val_chan_baseline + differences_baseline[Max_Abs_Idx_chan_baseline+PPSamp] + differences_baseline[Max_Abs_Idx_chan_baseline+(2*PPSamp)])#Create a triplet of the 2 peaks around our max peak 
                        Max_Abs_chan_baseline[swp][channel] = Triplets_baseline
                    
                    elif Max_Abs_Idx_chan_baseline+PPSamp >= len(differences_baseline):
                        Triplets_baseline = np.abs(Max_Abs_Val_chan_baseline + differences_baseline[Max_Abs_Idx_chan_baseline-PPSamp] + differences_baseline[Max_Abs_Idx_chan_baseline-(2*PPSamp)]) #Create a triplet of the 2 peaks around our max peak 
                        Max_Abs_chan_baseline[swp][channel] = Triplets_baseline
                    
                    else:
                        Triplets_baseline = np.abs(Max_Abs_Val_chan_baseline + differences_baseline[Max_Abs_Idx_chan_baseline+PPSamp] + differences_baseline[Max_Abs_Idx_chan_baseline-PPSamp]) #Create a triplet of the 2 peaks around our max peak 
                        # Max_Abs_chan_baseline[swp][channel] = np.max(differences_baseline)
                        Max_Abs_chan_baseline[swp][channel] = Triplets_baseline
                    
    
                    
    #calculate the std of the last 100 values of the corresponding sweep 
                # std_tail = np.std(swps[swp+1].signal[-100:, channel])                                              
                # stds[swp][channel] = std_tail

                
                #Check to n n+2 with n n+2 from the tail, last 100 entrances
                # differences_tail = np.abs(np.diff(swps[swp+1].signal[-int(durations[swp]*sampleRate):, channel], n = 2))  #Get the abs values of point n and n+2 in one seep for one channel
                # Tail_Max_Abs_chan[swp][channel] = np.max(differences_tail)
                ###
    
        ABS_STD_DIV = np.zeros((len(swps)-1, 32)) 
        ABS_STD_DIV_baseline = np.zeros((len(swps)-1, 32))
        for swp in range (len(Max_Abs_chan)):
            # abs difference to overnext sample during stimulation time 
            for channel in range(32):

                # ABS_STD_DIV[swp][channel] = Max_Abs_chan[swp][channel]/stds[swp][channel]
                ABS_STD_DIV[swp][channel] = Max_Abs_chan[swp][channel]#/stds[swp][channel]

     
                # ABS_STD_DIV_baseline[swp][channel] = Max_Abs_chan_baseline[swp][channel]/stds[swp][channel]
                ABS_STD_DIV_baseline[swp][channel] = Max_Abs_chan_baseline[swp][channel]#/stds[swp][channel]


        # std_columns = [column for column in df.columns if "std_chan" in column]
        # Max_Abs_columns = [column for column in df.columns if "Max_Abs_chan" in column]
        ABS_STD_DIV_columns = [column for column in df.columns if "ABS_STD_DIV" in column]
        ABS_STD_DIV_columns_baseline = [column for column in df.columns if "DIV_baseline" in column]
    
        
        # for idx, col in enumerate (std_columns):
        #     df[col] = stds[:,idx]
        # for idx, col in enumerate (Max_Abs_columns):
        #     df[col] = Max_Abs_chan[:,idx]
        
        for idx, col in enumerate (ABS_STD_DIV_columns):
            df[col] = ABS_STD_DIV[:,idx]
            
            
        #Doing the same for the baseline stuff:
        for idx, col in enumerate (ABS_STD_DIV_columns_baseline):
            df[col] = ABS_STD_DIV_baseline[:,idx] 
        
        #Save the csv file:
            #changed to new dataroot with freshly analyzed
        df.to_csv(config.FRESH_DIR+'/'+dataset.animDir[row]+'/'+filename, index=False)
        # df.to_csv(config.DATA_ROOT+'/'+dataset.animDir[row]+'/'+filename, index=False)
        #End of the std and Abs Val Saving part ############################################################################
        
        # apply AMUA filter
        # save AMUA results to AMUA.ephys
        print('Computing AMUA ...')
        fs=swps[1].sampleRate
        coefs=AMUAFilterCoeffs(fs)
        swpAMUA = []
        df = stimPar.copy()
        for idx, swp in enumerate(swpClean):   
            # apply AMUA filter and save AMUA results to AMUA.ephys
            # first swp possible shorter than AMUA padLen bring up error
            # fake first AMUA swp
            if idx == 0:
                fakeswp = ep.MCsweep(2000)
                fakeswp.signal = swpClean[0].signal
                swpAMUA.append(fakeswp)
            else:
                AMUA = calcAMUA(swp, coefs ,2000)
                swpAMUA.append(AMUA)
                sig = AMUA.signal
                dd = df.iloc[idx-1]['duration (s)']
                avgAMUA = sig[int(0.0005*AMUA.sampleRate):int(dd*AMUA.sampleRate), :].mean(axis = 0)
                for cc in range(32):
                    df.at[idx-1, 'AMUA_Avg_Chan' + str(cc)] = avgAMUA[cc]
                    ''''''
                    # nup=ep.nonUniqueParams(stimPar)
                    # parNames=list(nup.keys())
                    # p=ep.permutations(nup)   
                    # row = 23
                    # chanNum = 25
                    # stimIdx=ep.stimIndecesFor(stimPar, parNames, p[row,:])
                    # stim = p[row, :]
                    # duration = int(stim[0]*swpAMUA[1].sampleRate)
                    # plt.figure(figsize=(15, 7))
                    # for idx, swp in enumerate(swpClean[stimIdx+1]):
                    #     plt.subplot(5, 4, idx+1)
                    #     AMUA = calcAMUA(swp, coefs ,2000)
                    #     swpAMUA.append(AMUA)
                    #     plt.plot(AMUA.signal[:, chanNum])
                    #     plt.axvline(duration, color='r', linestyle='--')
                    #     plt.ylim(0e-6, 10e-6)
                    #     dd = df.iloc[idx-1]['duration (s)']
                    #     avgAMUA = AMUA.signal[:int(dd*AMUA.sampleRate), :].mean(axis = 0)
                    #     plt.title(f'swpIdx{stimIdx[idx]+1}, {idx}, avgAMUA:{avgAMUA[chanNum]:.2e}')
                    # plt.tight_layout()
                    ''''''
        print('Saving AMUA ...')
        writeEphysFileFromRow(dataset, row, swpAMUA, df, dataRoot = config.FRESH_DIR, writeAMUA=True)
        # writeEphysFileFromRow(dataset, row, swpAMUA, df, dataRoot = config.DATA_ROOT+'/', writeAMUA=True)
        print('...done with '+dataset.iloc[row].fileName+'\n')
#%% 
'''
Generate inventory of recoding sites for the study
The code below generated a CSV file that serves as inventory of recording sites from cleaned datafile. 
'''
dataset=pd.read_csv(config.ANALYSIS_ROOT+'/datafileInventory.csv')
penetrationNameList = []
channelList = []
Right_IC_List = []
cohort = []
for row in np.arange(len(dataset)):
    # if dataset.Cleaned[row] == 0:
    #     continue
    # else:
        penetrationNameList = penetrationNameList+[dataset.fileName[row]]*32
        channelList = channelList+np.arange(32).tolist()
        Right_IC_List = Right_IC_List + [dataset.rightIC[row]]*32
        cohort = cohort + [dataset.condition[row]]*32
df = pd.DataFrame({'fileName': penetrationNameList,'Channel': channelList, 'rightIC': Right_IC_List, 'cohort': cohort})
df.to_csv(config.ANALYSIS_ROOT+'/recordingsiteInventory.csv', index=False) 
  #%% 
'''
run Wilcoxon-test over 'datafileInventory.csv' to find out which channels are responsive
save results back to 'unitInventory.csv'
'''
# dataset=pd.read_csv(analysisRoot+'/datafileInventory_Debugging.csv')
dataset=pd.read_csv(analysisRoot+'/datafileInventory.csv')
# Here we pick the midline position as a somewhat arbitrary choice to see if the neurons are responseive at all
conditionToCheck = []
for key in paramsDict:
    values= paramsDict[key]
    print(f"\nFor {key}, these are the available values:")
    for value in values:
        print(f"{value}")
    selection = input(f"\nWhich {key} value do you want to choose?: ").strip()
    try:
        if isinstance(values[0], (int, float)):
            selection = type(values[0])(selection)
        if selection in values:
            conditionToCheck.append((key, selection))
            print(f"Selected {selection} for {key}")
    except ValueError:
        print("Please enter a valid number")
#%%    
# conditionToCheck=[('clickRate (Hz)',900), ('ITD (ms)',0), ('ILD (dB)',0)] 
dataRoot=config.FRESH_DIR+'/'

wilcoxonResultsList = []
selectionResultsList = []
print('Listing responsive units...')
for row in np.arange(len(dataset)):
    # if True:
    if row >= 59:
        filename = dataset.iloc[row]['fileName']
        print('Looking for responsive recording sites in '+filename+"...")
        swpsAMUA, stimPar = readEphysFileFromRow(dataset, row, loadAMUA = True)
        if isinstance(swpsAMUA, list):
            continue
        # else:
        results, selection = wilcoxonTestfor(swpsAMUA, stimPar, conditionToCheck)
        wilcoxonResultsList = wilcoxonResultsList+results
        selectionResultsList = selectionResultsList+selection
# print('... Done.')
# df = pd.read_csv(analysisRoot+'/recordingsiteInventoryV2.csv')
# df = pd.read_csv(config.ANALYSIS_ROOT+'/Salt_Beaker_CSV_Files'+'/recordingsiteInventory_Salt_Beaker_VE.csv')
# df=pd.read_csv(analysisRoot+'/recordingsiteInventory_Debugging.csv')
df=pd.read_csv(analysisRoot+'/recordingsiteInventory_ITD+.csv')

df['wilcoxon-pValue'] = wilcoxonResultsList
df['Responsive'] = selectionResultsList
# df.to_csv(analysisRoot+'/'+'recordingsiteInventoryV2.csv', index=False)   
df.to_csv(config.ANALYSIS_ROOT+'/recordingsiteInventory_ITD+.csv', index=False) 



#%%

'''
Come up with the criterion to exclude the data with leftover artefact --> read in _Clean Data and get columns with ABS/STD values
the plotting functions below are to checkwhich std should be used as criterion... 
'''

dataset=pd.read_csv(analysisRoot+'/datafileInventory.csv')

# for row in np.arange(len(dataset)): # set through file list
    #load ephys data
    # if row >28 and row < 59: #  and row < 30: # for debugging, set range of rows here
    # if True: # otherwise run everything
row = 86
# row = 90
print('loading file '+ dataset.iloc[row].fileName+'...')
swps, stimPar = readEphysFileFromRow(dataset, row, stimVarParams, loadClean =True)
    

abs_std_columns = [col for col in stimPar.columns if 'ABS_STD_DIV' in col]
abs_std_columns_baseline = [col for col in stimPar.columns if 'DIV_baseline' in col]

# Create a new DataFrame with only ABS_STD columns
df_abs_std = stimPar[abs_std_columns]
df_abs_std_baseline = stimPar[abs_std_columns_baseline]   

#Plot the histogram and find ou the 95th percentile:
# chan = 25
for idx, column in enumerate(df_abs_std.columns):
    # print(df_abs_std[column])
    plt.hist(df_abs_std[column], bins=30, density=True, alpha=0.6, color='blue', edgecolor='black')


percentile_95th = np.zeros(32)          # holds the values of the 95th percentile for each channel 
for idx, column in enumerate(df_abs_std_baseline.columns):
    # print(df_abs_std[column])
    #For the baseline -> find out the 95th percentile value for each channel --> if the df_abs_std value is bigger than 2x this value its considered "not clean". 
    percentile_value = np.percentile (df_abs_std_baseline[column], 95)
    percentile_95th[idx] = percentile_value
    plt.hist(df_abs_std_baseline[column], bins=30, density=True, alpha=0.6, color='red', edgecolor='black')
    
plt.title('Distribution Plot of the max ABS value divided by STD value')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()


Exceeding_95th_percentile = []
#Find the indices in the df_abs_std that exceed the 2x 95th percentile values:
for idx, column in enumerate(df_abs_std.columns):
    indices = np.where(df_abs_std[column] > 2 * percentile_95th[idx])[0]
    Exceeding_95th_percentile.append(indices)
    
    

# chan = 0
for i in range (len(Exceeding_95th_percentile)):
    print (i)
    plt.figure()
    for ii in Exceeding_95th_percentile[i] :
        plt.plot(swpClean[ii+1].signal[:,i], label = ii)
        plt.legend()

# plt.plot(swps[17].signal[:,31])    
# plt.plot(swps[130].signal[:,31]) 
# plt.plot(swps[275].signal[:,31]) 
# plt.plot(swps[324].signal[:,31])   
# plt.plot(swps[787].signal[:,31])    
 
    
    

# Plotting single sweeps:
max_value = df_abs_std.values.max()
max_idx = np.unravel_index(df_abs_std.values.argmax(), df_abs_std.shape)
max_row, max_col = max_idx

plt.figure()
plt.plot(swps[int(max_row+1)].signal[:,int(max_col)])

plt.figure()
plt.plot(swps[1].signal[:,0])

# #z-score
# z_scores = zscore(Teiler, axis = None)
# outliers = np.argwhere(np.abs(z_scores) > 2)
# for i in range(32):
#     plt.hist(z_scores[:,i], bins=30, density=True, alpha=0.6, color='blue', edgecolor='black')
    
    
#Top 5%
#threshold = np.percentile(Teiler, 95)  # 95th percentile
#Find the indices of values >= threshold
# indices = np.argwhere(Teiler >= threshold)
#Extract the values and indices
# top_values = Teiler[Teiler >= threshold]
# top_indices = indices --> indices of the top 5% values
#--> plot: plt.plot(swps[top_indices[0][0]].signal[:,top_indices[0][1]])

#%%
'''
baseline check, take baseline 20ms before the recording
'''
dataset=pd.read_csv(analysisRoot+'/datafileInventory_V3.csv')
dataset = dataset[dataset['condition']=='NH']
for ii in range(len(dataset)):
    penetrationinfo = dataset.iloc[ii]
    if penetrationinfo['Cleaned'] == 0:
        continue
    fname = penetrationinfo['fileName']
    anim = penetrationinfo['animDir']
    swpsClean, stimClean = readEphysFileFromRow(dataset, ii, stimVarParams, loadClean=True)
    stimPar = pd.read_csv(dataRoot+'/'+anim+'/'+fname[:-5]+'csv')
    stimPar['baseline'] = ''
    sr = swpsClean[0].sampleRate
    sn = int(sr*0.02)
    for jj, swp in enumerate(swpsClean[:-1]):
        avg = np.mean(swp.signal[:-488])
        stimPar.at[jj, 'baseline'] = avg
    
    stimPar.to_csv(dataRoot+'/'+anim+'/'+fname[:-5]+'csv', index=False) 

