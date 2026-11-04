#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:44:46 2024

@author: shiyi
"""
import pandas as pd 
from sys import path
path.append('/home/colliculus/behaviourBoxes/software/ratCageProgramsV3/')
import RZ2ephys as ep
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import re

analysisRoot='/ephys/2/0_ENVvsPT/Analysis'
figureRoot = '/ephys/2/0_ENVvsPT/Figure'
dataRoot='/ephys/2/0_ENVvsPT/Data'

dataFiles = pd.read_csv(analysisRoot+'/datafileInventory.csv')
recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventory.csv')
leftoverArt = pd.read_csv(analysisRoot+'/LeftoverArtefact.csv', low_memory=False)
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
    return sub_recordingSites

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
'''
Visulization tool
'''
def prepare4X8Plot():
    fig = plt.figure(figsize = (20, 10))
    fig.set_tight_layout(True)
    gs = gridspec(4, 8)
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
VE distribution plot
'''
corhorts = dataFiles.condition.unique()

for corhNum, corhort in enumerate(corhorts):
    fig, axs = prepare2X3Plot()
    anims = dataFiles[dataFiles['condition'] == corhort]['animDir'].unique()
    unitNum = 0
    for animNum, anim in enumerate(anims):
        filenames = dataFiles[(dataFiles['animDir'] == anim) & (dataFiles['Cleaned'] == 1)]['fileName']
        for penNum, filename in enumerate(filenames):
            sub_recordingSites = recordingSites[(recordingSites['fileName'] == filename) & (recordingSites['Responsive'] == 1)]
            unitNum+=len(sub_recordingSites)
            scatterText = 'anim'+str(animNum)+'pen'+str(penNum)
            fig = plot2X3Scatter(sub_recordingSites, fig, axs, scatterText)
        fig.text(0, 0.5, 'Variance explained by PT_ITD (%)', va='center', rotation=-90, fontsize = 14)
        fig.text(0.38, 0.053, 'Variance explained by ENV_ITD (%)', va='center', fontsize = 14)
        fig.suptitle(corhort+'_ResponsiveUnit#'+str(unitNum), fontsize = 14)
        plt.savefig('/ephys/2/0_ENVvsPT/Figure/Result_ScatterPlot/'+corhort)
#%%
'''

'''
corhort = 'NH'
animNum = 5
penNum = 3
chanNum = 5
duration = 0.2
clickRate = 900

anim = dataFiles[(dataFiles['condition'] == corhort)]['animDir'].unique()[animNum]
filename = dataFiles[(dataFiles['animDir'] == anim) & (dataFiles['Cleaned'] == 1)].iloc[penNum]['fileName']
swpsClean, stimPar=ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-6]+'_Clean')
swpsAMUA, stimPar=ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-6]+'_AMUA')

#%% 3X3 plot
conditions = [('duration (s)',duration), ('clickRate (Hz)',clickRate)]
VEheader, significanceHeader = prepareHeader(recordingSites.columns, str(duration), str(clickRate))
df = recordingSites[(recordingSites['fileName'] == filename) & (recordingSites['Channel'] == chanNum)]
header, significanceHeader = prepareHeader(df, str(duration), str(clickRate))
xy = df[header]
fig = plot3X3plot(swpsClean, conditions, chanNum, allRepeats=True)
fig.suptitle(corhort+' VEpt'+str('{:.2f}'.format(xy.iloc[0][VEheader[0]]*100))+'% VEenv'+str('{:.2f}'.format(xy.iloc[0][VEheader[1]]*100))+'%')
fig = plot3X3plot(swpsAMUA, conditions, chanNum, avgRepeats=True)
fig.suptitle(corhort+' VEpt'+str('{:.2f}'.format(xy.iloc[0][VEheader[0]]*100))+'% VEenv'+str('{:.2f}'.format(xy.iloc[0][VEheader[1]]*100))+'%')
#%% example AMUA trace (-0.1, 0, 0.1)
cc = 10
amua_array = contra_amua
amua_plot = amua_array[cc, 0, 1, :]
amua_plot = filtfilt(bLow,aLow, amua_plot, axis=2, padlen=100)
amua_max = np.max(amua_plot)
amua_normalize = amua_plot/amua_max
amua_svd = np.std(amua_normalize, axis = -1)
amua_plot = np.mean(amua_normalize, -1)

amua_trace = amua_plot[:, :, int(t_start*Fs_down):int(t_end*Fs_down)]
amua_svd = amua_svd[:, :, int(t_start*Fs_down):int(t_end*Fs_down)]
amua_trace = amua_trace-np.mean(amua_trace[:, :, :6])

fig1 = plt.figure(figsize=(12, 8),constrained_layout=True)
gs0 = gridspec.GridSpec(1, 1)
gs0.update(left=0.1, right=0.9, bottom = 0.1, top = 0.9, wspace=0, hspace=0)
ax0 = fig1.add_subplot(gs0[0, 0])
ax0.set_ylabel('Normalized AMUA', rotation =  'vertical', fontsize = 20)
ax0.set_xlabel('Peristimulus time(ms)', fontsize = 20)
ax0.tick_params(axis='both', which='major', labelsize=20)
for ii in range(3):
    ax0.plot(t, amua_trace[ii, ii, :])
    ax0.fill_between(t, (amua_trace[ii, ii, :] - amua_svd[ii, ii, :].T), (amua_trace[ii, ii, :] + amua_svd[ii, ii, :].T), alpha=0.15)
    
ax0.legend(['ITDs Negative', 'ITDs 0', 'ITDs positive'], fontsize = 20)
ax0.axvline(50, c='tab:red', ls='--')
