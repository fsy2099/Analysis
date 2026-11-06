#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 16:42:18 2026

@author: shiyi

plot example for triplet illustration
"""
import config  # data/figure folders, see config.py
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import re
import statistics
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import numpy as np
from matplotlib.gridspec import GridSpec
from sys import path
path.append(config.EPHYS_LIB+'/')
import RZ2ephys as ep
from tqdm import tqdm
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import copy


'''
################################################################################################################################
This code is a individual code to analyse the AMUA results of the recorded cue trading data (Step 3)
################################################################################################################################
'''



'''
write down the important conditions before starting the pipeline- so adjust these parameters and tell the program which data is going to be analyzed. Call this variable: "Data_conds"
makes life maybe easier. 
Cue Trading: [('clickRate (Hz)',1, 100, 900]),[ ('ITD (ms)',[-120, -80, -40, 0, 40, 80, 120]), ('ILD',[-4, -1, 0, 1, 4])]
Env Vs Pt:  [('duration (s)',[0.01, 0.05, 0.2]), ('clickRate (Hz)',[900, 4500]), ('ITD (ms)',[-0.1, 0, 0.1]), ('env ITD (ms)',[-0.1, 0, 0.1])]
Weighting: ?



'''

# specify the stim params that were permuted to generate different stimulus conditions
stimVarParams = ['duration (s)', 'clickRate (Hz)', 'ITD (ms)', 'env ITD (ms)']




dataRoot = config.DATA_ROOT
analysisRoot = config.ANALYSIS_ROOT
figureRoot = config.FIGURE_ROOT


'''
Read in & write in
'''


def readEphysFileFromRow(dataset, row, dataRoot = dataRoot, loadOrig=False, loadAMUA=False, loadClean=False):
    filename = dataset.fileName[row]
    if filename.endswith('.ephys'):
        filename = filename[:-6]
    if loadOrig == True:
        filename = filename
    if loadAMUA == True:
        filename = filename+'_AMUA'
    if loadClean == True:
        filename = filename+'_Clean'
    # if no 'Cleaned' column in csv file, read in all row
    if not any('Cleaned' in col for col in dataset.columns):
        swps, stimPar = ep.readEphysFile(
            dataRoot+'/'+dataset.animDir[row]+'/'+filename)
        # swps, stimPar = checkSwpsSize(swps, stimPar)
        print(filename)
    # if exist 'Cleaned' column in csv file, only read in row labed as '1' in cleaned
    if any('Cleaned' in col for col in dataset.columns):
        if dataset.Cleaned[row] == 1:
            swps, stimPar = ep.readEphysFile(
                dataRoot+'/'+dataset.animDir[row]+'/'+filename)
            # swps, stimPar = checkSwpsSize(swps, stimPar)
        if dataset.Cleaned[row] == 0:
            swps = []
            stimPar = []
    return swps, stimPar


def writeEphysFileFromRow(dataset, row, swps, stimPar, writeAMUA=False, writeClean=False):
    if writeAMUA == True:
        filename = dataset.fileName[row][:-6]+'_AMUA'
    if writeClean == True:
        filename = dataset.fileName[row][:-6]+'_Clean'
    ep.writeEphysFile(
        dataRoot+'/'+dataset.animDir[row]+'/'+filename, swps, stim=stimPar)


def writeCsvFileFromRow(dataset, row, writeWilcoxon=False):
    if writeWilcoxon == True:
        filename = dataset.fileName[row][:-6]+'.csv'
    dataset.to_csv(dataRoot+'/'+dataset.animDir[row]+'/'+filename, index=False)

'''
some useful function
'''


def findSweepsFor(conditions, stimPar):
    # returns a list of indeces into for the sweeps that satisfy the conditions.
    # conditions should be a list,  of pairs (paramName, paramValue)
    # For now e only do equality, and logical AND, ie we look for sweeps
    # satisfying (paramName1==paramValue1) AND (paramName2==paramValue2) ...
    if not type(conditions) == list:
        raise Exception(
            "conditions must be a list of pairs, eg [('duration (s)',0.01),('clickRate (Hz)',900)]")
    acond = conditions.pop()
    select = stimPar[acond[0]] == acond[1]
    while len(conditions) > 0:
        acond = conditions.pop()
        select = np.logical_and(select, stimPar[acond[0]] == acond[1])
    swpIdx = np.where(select)[0]
    return swpIdx + 1


def findConditionsForswp(swp, stimPar):
    aStim = stimPar.iloc[swp-1]
    conds = []
    for aCond in stimVarParams:
        # for aCond in aStim.keys():
        # if aCond.find('imp_')==0:
        #     continue
        # if aCond.find('timeStamp')==0:
        #     continue
        conds.append((aCond, f'{aStim[aCond]}'))
        # print(aCond)
    return conds

def poolChannel(swps, channelNum):
    # pool out all repeats for the given channel
    alen = np.zeros([len(swps)])
    for ss in range(len(swps)):
        alen[ss] = swps[ss].signal.shape[0]
    minLen = int(np.min(alen))
    signal = []
    for swp in swps:
        signal.append(swp.signal[:minLen, channelNum])
    signal = np.array(signal).T
    return signal


def findRecordingSite(dataFiles, row):
    filename = dataFiles['fileName'][row]
    sub_recordingSites = recordingSites[recordingSites['fileName'] == filename]
    # sub_recordingSites = sub_recordingSites[sub_recordingSites['Responsive'] == 1]
    return sub_recordingSites


def writeValueinto(df, idx, V1, V2, Vp1, Vp2, keyName):
    df = checkKeyNamein(df, keyName)
    df.at[idx, keyName+'_VEpt'] = V1
    df.at[idx, keyName+'_VEenv'] = V2
    df.at[idx, '(pValue)'+keyName+'_VEpt'] = Vp1
    df.at[idx, '(pValue)'+keyName+'_VEenv'] = Vp2
    if V1 > Vp1:
        df.at[idx, keyName+'_VEpt_significance'] = 1
    else:
        df.at[idx, keyName+'_VEpt_significance'] = 0
    if V2 > Vp2:
        df.at[idx, keyName+'_VEenv_significance'] = 1
    else:
        df.at[idx, keyName+'_VEenv_significance'] = 0
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


'''
Wilcoxon Test
'''


def wilcoxonTestfor(swps, stimPar, conditionToCheck):
    swpIdx = findSweepsFor(conditionToCheck.copy(), stimPar)
    swpTest = swps[swpIdx]
    sampleRate = swpTest[0].sampleRate
    nchans = swpTest[0].signal.shape[1]
    # duration = conditionToCheck[0][1]
    duration = stimPar.iloc[swpIdx[0]]['duration (s)']
    selectionArray = []
    resultsArray = []
    for cc in range(nchans):
        signal = poolChannel(swpTest, cc)
        responseWinAvg = signal[:int(duration*sampleRate), :].mean(axis=0)
        baselineWinAvg = signal[-int(duration*sampleRate):, :].mean(axis=0)
        results = stats.wilcoxon(responseWinAvg, baselineWinAvg)
        resultsArray.append(results[1])
        if results[1] < 0.05:
            selectionArray.append(1)
        else:
            selectionArray.append(0)
    return resultsArray, selectionArray


'''
variance explained ratio calculation     
'''


def calcVarianceExplained(df):
    ss_t = ssTotalCalc(df)
    # model = ols('AMUA ~ C(PT_ITD)+C(ENV_ITD)+C(PT_ITD):C(ENV_ITD)',data=df).fit()
    # result = sm.stats.anova_lm(model, type=1)
    model = ols('AMUA ~ C(PT_ITD)+C(ENV_ITD)',data=df).fit()
    result = sm.stats.anova_lm(model, type=2)
    print(result)
    ss_explainedTotal = result.iloc[0, 1]+result.iloc[1, 1]
    # ss_pt = result.iloc[0, 1]/ss_explainedTotal
    # ss_env = result.iloc[1, 1]/ss_explainedTotal
    ss_pt = result.iloc[0, 1]/ss_t
    ss_env = result.iloc[1, 1]/ss_t
    ss_pt_p = result.iloc[0, 4]
    ss_env_p = result.iloc[1, 4]
    return ss_pt, ss_env, ss_pt_p, ss_env_p


def ssTotalCalc(df):
    totalMean = df['AMUA'].mean()
    ss_t = sum((df['AMUA']-totalMean)**2)
    return ss_t


def permutationTestfor(df, num=1000):
    ss_pt_array = []
    ss_env_array = []
    print('start permuting...')
    df = df.reset_index(drop=True)
    for tt in range(num):
        shuffled_index = np.random.permutation(df.index)
        df['AMUA'] = df['AMUA'].iloc[shuffled_index].values
        ss_pt, ss_env, p1, p2 = calcVarianceExplained(df)
        ss_pt_array.append(ss_pt)
        ss_env_array.append(ss_env)
    ss_pt_p = np.percentile(np.array(ss_pt_array), 95)
    ss_env_p = np.percentile(np.array(ss_env_array), 95)
    return ss_pt_p, ss_env_p


'''
Visulization tool
'''


def prepare4X8Plot():
    fig = plt.figure(figsize=(20, 10))
    fig.set_tight_layout(True)
    gs = GridSpec(4, 8)
    axs = []
    for ii in range(32):
        x = ii % 8
        y = ii//8
        ax = fig.add_subplot(gs[y, x])
        ax.set_title('Ch'+str(ii))
        axs.append(ax)
    return fig, axs


def plot32Channel(swps, stimPar, conditions, allRepeats=False, avgRepeats=False, ncols=8, nrows=4):
    sampleRate = swps[1].sampleRate
    Duration = conditions[0][1]
    t = np.arange(0, Duration+0.05, 1/sampleRate)
    nchans = swps[1].signal.shape[1]
    swpIdx = findSweepsFor(conditions.copy(), stimPar)
    swpPlot = swps[swpIdx]

    fig, axs = prepare4X8Plot()
    for cc in range(nchans):
        ax = axs[cc]
        signal = poolChannel(swpPlot, cc)
        if avgRepeats:
            signal = signal[:len(t), :].mean(axis=1)
            ax.plot(t, signal)
        if allRepeats:
            signal = (signal-signal[0, :])[:len(t)]
            ax.plot(t, signal)

        ax.axvline(x=Duration, color='k', linestyle='--')
        if not np.mod(cc, ncols) == 0:
            plt.yticks([])
        if not cc >= nchans-ncols:
            plt.xticks([])
    return fig


def prepare3X3Plot(itd=[-0.1, 0, 0.1]):
    fig = plt.figure(figsize=(8, 8), constrained_layout=False)
    gs = fig.add_gridspec(3, 3)
    gs.update(left=0.05, right=0.95, bottom=0.08,
              top=0.92, wspace=0.15, hspace=0.3)
    axs = []
    for ii in range(9):
        xx = ii % 3
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


def plot3X3plot(swps, stimPar, sub_conditions, chanNum, itd=[-0.1, 0, 0.1], allRepeats=False, avgRepeats=False,):
    sampleRate = swps[1].sampleRate
    duration = sub_conditions[0][1]
    clickRate = sub_conditions[1][1]
    t = np.arange(0, duration+0.05, 1/sampleRate)
    fig, axs = prepare3X3Plot()
    for ii in range(9):

        xx = ii % 3
        yy = ii//3
        pt_itd = itd[yy]
        env_itd = itd[xx]
        ax = axs[ii]

        conditions = [('duration (s)', duration), ('clickRate (Hz)',
                                                   clickRate), ('ITD (ms)', pt_itd), ('env ITD (ms)', env_itd)]
        swpIdx = findSweepsFor(conditions.copy(), stimPar)
        signal = poolChannel(swps[swpIdx], chanNum)
        if allRepeats == True:
            signal = (signal-signal[0, :])[:len(t), :]
        if avgRepeats == True:
            signal = signal.mean(axis=1)[:len(t)]
        ax.plot(t, signal)
        ax.axvline(x=duration, color='k', linestyle='--')
    return fig


def prepare2X3Plot(durs=[0.01, 0.05, 0.2], pps=[900, 4500]):
    fig = plt.figure(figsize=(12, 8), constrained_layout=False)
    gs = fig.add_gridspec(2, 3)
    gs.update(left=0.05, right=0.98, bottom=0.12,
              top=0.92, wspace=0.1, hspace=0.15)
    axs = []
    for ii in range(6):
        xx = ii % 3
        yy = ii//3
        rate = str(pps[yy])
        dur = str(durs[xx])
        ax = fig.add_subplot(gs[yy, xx])
        # ax.set_xlim(-5, 105)
        # ax.set_ylim(-5, 105)
        ax.set_title(f'{float(dur)*1000:.0f} ms, {rate} pps')
        axs.append(ax)
    return fig, axs


'''
VE distribution plot
'''


def plotScatter(xy, header, ax, text, color):
# header = prepareHeader(sub_recordingSites, ii)
# xy = prepareScatter(sub_recordingSites, header, jj)
    x = xy[header[2]]
    y = xy[header[1]]
    channel = xy[header[0]]
    # for i in range(x.shape[0]):
    #     ax.text(x.iloc[i]*100, y.iloc[i]*100, text+'_ch'+str(channel.iloc[i]), alpha = 0.4, ha = 'center', rotation = 45)
    # print(x.shape)
    ax.scatter(x*100, y*100, color=color, facecolors='none', hatch='///')
    ax.tick_params(axis = 'both', labelsize = 14)
    ax.set_xlim(-5, 100)
    ax.set_ylim(-5, 100)
    
def Header(sub_recordingSites, ii, durs=[0.01, 0.05, 0.2], pps=[900, 4500]):
    xx = ii % 3
    yy = ii//3
    rate = str(pps[yy])
    dur = str(durs[xx])
    columns = sub_recordingSites.columns
    VEheader,  significanceHeader = prepareHeader(columns, rate, dur)
    return VEheader, significanceHeader


def prepareHeader(columns, fac1, fac2):
    VEheader = [value for value in columns if 'VE' in value and fac1 in value and fac2 in value and '_significance' not in value and 'pValue' not in value or 'Channel' in value]
    significanceHeader = [
        value for value in columns if 'VE' in value and fac1 in value and fac2 in value and '_significance' in value]
    return VEheader, significanceHeader


def plot2X3Scatter(sub_recordingSites, sub_baselinerecordingSites, fig, axs, text, durs=[0.01, 0.05, 0.2], pps=[900, 4500], colors=None, type=None):
# fig, gs_scatter = prepare2X3Plot()
    if type == 'significancy':
        colors = ['tab:blue', 'tab:red', 'purple', 'grey']
        Label = [[1, 0], [0, 1], [1, 1], [0, 0]]
        for ii in range(6): # loop 2 clickrate*3duration
            # print(ii)
            ax = axs[ii]    
            VEheader, significanceHeader = Header(sub_recordingSites, ii)
            # ''''''
            df = sub_baselinerecordingSites[(sub_baselinerecordingSites[significanceHeader[0]] == 0) & (sub_baselinerecordingSites[significanceHeader[1]] == 0)] # select out unit non significant to pt or env at baseline
            keepidx = df.index
            # ''''''
            for jj in range(4):
                # print(jj)
                sub_subrecordingSites = sub_recordingSites.loc[keepidx]
                # sub_subrecordingSites = sub_recordingSites
                # df = sub_recordingSites[(sub_recordingSites[significanceHeader[0]] == Label[jj][0]) & (sub_recordingSites[significanceHeader[1]] == Label[jj][1])]
                df = sub_subrecordingSites[(sub_subrecordingSites[significanceHeader[0]] == Label[jj][0]) & (sub_subrecordingSites[significanceHeader[1]] == Label[jj][1])]
                xy = df[VEheader]
                # print(len(xy))
                plotScatter(xy, VEheader, ax, text, color=colors[jj])
    if type == 'penetration':
        for ii in range(6): # loop 2 clickrate3duration
            ax = axs[ii]
            VEheader, significanceHeader = Header(sub_recordingSites, ii)
            xy = sub_recordingSites[VEheader]
            plotScatter(xy, VEheader, ax, text, color=colors)
        if ii == 0:
            line = Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='none', markeredgecolor=colors,
                          abel=text, markersize=10, markeredgewidth=1.5)
            fig.legend_elements.append(line)
            # Place legend outside all subplots
            # fig.legend(handles=fig.legend_elements,
            # loc='center left',
            # bbox_to_anchor=(1.05, 0.5),
            # fontsize=10)

    return fig


# Function to check which sweeps have left over artefact:
'''
Previously used in the VE calculation to exclude the data with leftover artefact and exclude if more than 1/4 of the sweeps with the same conds are kicked out
'''

def LeftOverArtefact(nup, p, channel, stimParCLEAN):

    # Remove keys for further analysis if there are residual artefacts in more than 1/4 of the sweeps.
    # Holds all the keys that need to be removed for this channel

    Headername = f'ABS_STD_DIV_chan{channel}'  # Ensure exact match
    abs_std_columns = [
        col for col in stimParCLEAN.columns if col == Headername]

    Headername_baseline = f'DIV_baseline{channel}'  # Ensure exact match
    abs_std_columns_baseline = [
        col for col in stimParCLEAN.columns if col == Headername_baseline]

    # Create a new DataFrame with only ABS_STD columns
    df_abs_std = stimParCLEAN[abs_std_columns]
    df_abs_std_baseline = stimParCLEAN[abs_std_columns_baseline]

    # Plot the histogram if needed:
    # plt.hist(df_abs_std, bins=30, density=True, alpha=0.6, color='blue', edgecolor='black')
    # plt.hist(df_abs_std_baseline, bins=30, density=True, alpha=0.6, color='red', edgecolor='black')
    # plt.title('Distribution Plot of the max ABS value divided by STD value')
    # plt.xlabel('Values')
    # plt.ylabel('Density')

    # Calculate the 95th percentile of the baseline
    percentile_value = np.percentile(df_abs_std_baseline, 95)
    leftoverart = np.where(df_abs_std > 2 * percentile_value)[0]
    # leftoverart = leftoverart + 1
    # leftoverart = leftoverart -1

    if len(leftoverart) == 0:
        remove_keys = np.empty(0, dtype=object)
        # return remove_keys

    elif len(leftoverart) > 0:
        Artefact_check = []  # Holds combinations with too much artefact
        for ind, stim in enumerate(p):
            idxs_prev = ep.stimIndecesFor(stimParCLEAN, list(nup.keys()), stim)
        # check if there is residual artefact:
            idxs = np.array(
                [item for item in idxs_prev if item not in leftoverart])
            # print(len(idxs))
        # Check if left over sweeps are at least 2/3 of the trials presented.
            if len(idxs) < 0.75 * len(idxs_prev):
                print('Too much leftover aretefact in combination: ', stim)
                Artefact_check.append(stim)

#      #After the loop below: holds all the values fo keys that need to be removed:
        remove_keys = np.empty(len(Artefact_check), dtype=object)
        for i in range(len(Artefact_check)):
            remove_keys[i] = ('Dur'+str(Artefact_check[i][0]) +'_Rate'+str(Artefact_check[i][1]))
        remove_keys = np.unique(remove_keys)
#         # print('Too much leftover aretefact in combination: ', remove_keys)
    return remove_keys


def remove_single_ARTsweeps(idxs_prev, channel, stimParCLEAN):
    Headername = f'ABS_STD_DIV_chan{channel}'  # Ensure exact match
    abs_std_columns = [col for col in stimParCLEAN.columns if col == Headername]

    Headername_baseline = f'DIV_baseline{channel}'  # Ensure exact match
    abs_std_columns_baseline = [col for col in stimParCLEAN.columns if col == Headername_baseline]

    # Create a new DataFrame with only ABS_STD columns
    df_abs_std = stimParCLEAN[abs_std_columns]
    df_abs_std_baseline = stimParCLEAN[abs_std_columns_baseline]

    # Plot the histogram if needed:
    # plt.hist(df_abs_std, bins=30, density=True, alpha=0.6, color='blue', edgecolor='black')
    # plt.hist(df_abs_std_baseline, bins=30, density=True, alpha=0.6, color='red', edgecolor='black')
    # plt.title('Distribution Plot of the max ABS value divided by STD value')
    # plt.xlabel('Values')
    # plt.ylabel('Density')

    # Calculate the 95th percentile of the baseline
    percentile_value = np.percentile(df_abs_std_baseline, 95)
    leftoverart = np.where(df_abs_std > 2 * percentile_value)[0]
    # leftoverart = leftoverart + 1
    # leftoverart = leftoverart -1

    idxs = [item for item in idxs_prev if item not in leftoverart]
    idxs = np.array(idxs, dtype=int)

    return idxs
#%%
'''
variance explained calculation
read in datafileInventory.csv and recordingsiteInventory.csv
'''
# dataFiles = pd.read_csv(analysisRoot+'/Salt_Beaker_CSV_Files/datafileInventory_Salt_Beaker_Agar.csv')
# recordingSites = pd.read_csv(analysisRoot+'/Salt_Beaker_CSV_Files/recordingsiteInventory_Salt_Beaker_Agar.csv')

dataFiles=pd.read_csv(config.ANALYSIS_ROOT+'/datafileInventory.csv')
recordingSites = pd.read_csv(config.ANALYSIS_ROOT+'/recordingsiteInventory.csv')
#%%
# checkParams = [4500.0, 0.01]

# corhorts = dataFiles.condition.unique()
# # colors = plt.cm.tab20(np.linspace(0, 1, 40))
# for corhNum, corhort in enumerate(corhorts):
#     if corhNum == 0:
#         # if True:
#         # fig, axs = prepare2X3Plot()
#         # if not hasattr(fig, 'legend_elements'):
#         #     fig.legend_elements = []
#         anims = dataFiles[dataFiles['condition']== corhort]['animDir'].unique()
#         print(anims)
#         unitNum = 0
#         # coloridx = 0
#         for animNum, anim in enumerate(anims):
#             # if animNum == 0:
#             if True:
#                 # coloridx = 0
#                 # fig, axs = prepare2X3Plot()
#                 # if not hasattr(fig, 'legend_elements'):
#                 # fig.legend_elements = []
#                 filenames = dataFiles[(dataFiles['animDir'] == anim) & (dataFiles['Cleaned'] == 1)]['fileName']
#                 for penNum, filename in enumerate(filenames):
#                     # if penNum == 0:
#                     if True:
#                         # if penNum in [0, 1]:
#                         sub_recordingSites = recordingSites[(recordingSites['fileName'] == filename)]
#                         # sub_recordingSites = baselinerecordingSites[baselinerecordingSites['fileName'] == filename]
#                         key = f'Dur{str(checkParams[1])}_Rate{str(checkParams[0])}_VEpt'
#                         i = np.where((sub_recordingSites[key] > 0.2) & (sub_recordingSites[f'{key}_significance'] == 1))[0]
#                         targetidx = sub_recordingSites.index[i]
#                         target = sub_recordingSites.iloc[i]
#                         chs = target.Channel
#                         if not chs.empty:
#                             print(f'{filename},\ncorhNum:{corhNum},animNum:{animNum},penNum:{penNum}:\n{chs.values}')
#%%
dataRoot = config.FRESH_DIR
corhort = 'NH'
animNum = 0
penNum = 0
chanNum = 16
duration = 0.05
clickRate = 900

anim = dataFiles[(dataFiles['condition'] == corhort)]['animDir'].unique()[animNum]
filename = dataFiles[(dataFiles['animDir'] == anim) & (dataFiles['Cleaned'] == 1)].iloc[penNum]['fileName']
# anim = '2022_01_25'
# filename = '20220125_ENVvsPT_5_P3_1.ephys'
print(filename)
swps, stimPar = ep.readEphysFile(config.DATA_ROOT +'/'+anim+'/'+filename[:-6])
swpsClean, stimParCLEAN = ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-6]+'_Clean')
# swpsAMUA, stimPar = ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-6]+'_AMUA')
#%%
conditions=[('duration (s)',duration), ('clickRate (Hz)',clickRate), ('ITD (ms)',0), ('env ITD (ms)',0)]
swps_idxs =findSweepsFor(conditions.copy(), stimPar)

swpsOut = copy.deepcopy(swps)
ep.artefactAlign(swpsOut, stimPar)
# signal = poolChannel(swpsOut[swps_idxs], chanNum)
# plt.figure(); plt.plot(signal)
ep.bandpassAllSweepsInFrequencyDomain(swpsOut)
# signal = poolChannel(swpsOut[swps_idxs], chanNum)
# plt.figure(); plt.plot(signal)
ep.correct_regression(swpsOut, stimPar)
# signal = poolChannel(swpsOut[swps_idxs], chanNum)
# plt.figure(); plt.plot(signal)
ep.artefactReject(swpsOut, stimPar)
# signal = poolChannel(swpsOut[swps_idxs], chanNum)
# plt.figure(); plt.plot(signal, 'o-')
#%%
# 设置色盲友好的颜色
# 使用蓝色和橙色的配色方案

# BASELINE_COLOR = '#9467bd'
# BASELINE_COLOR = '#bcbd22'
BASELINE_COLOR = '#17becf' 
RESPONSE_COLOR = '#ff7f0e'  # 橙色
BASELINE_ALPHA = 0.6
RESPONSE_ALPHA = 0.6

swp = swps[swps_idxs[1]]
swp_clean = swpsOut[swps_idxs[1]]
stim = stimPar.iloc[swps_idxs[1]-1]
# 获取参数
sampleRate = swp.sampleRate
duration = stim["duration (s)"]  
rate = stim["clickRate (Hz)"]    
baseline_val = stim["baseline"]  

# 计算PPSamp
if rate == 1:
    PPSamp = 1
else:
    PPSamp = int(np.round(sampleRate/rate))

# 获取原始和clean信号
full_signal = swp.signal[:, chanNum]
full_signal_clean = swp_clean.signal[:, chanNum]

# 确保两个信号长度相同
min_length = min(len(full_signal), len(full_signal_clean))
full_signal = full_signal[:min_length]
full_signal_clean = full_signal_clean[:min_length]

# 获取刺激期间和基线期间的clean信号
stim_end = int(duration*sampleRate)
baseline_start = stim_end + 200  
stim_signal_clean = full_signal_clean[:stim_end]
baseline_signal_clean = full_signal_clean[baseline_start:]

# 创建时间轴
full_time = np.arange(min_length) / sampleRate
stim_time = np.arange(len(stim_signal_clean)) / sampleRate
baseline_time = np.arange(len(baseline_signal_clean)) / sampleRate + (baseline_start / sampleRate)

# 计算clean信号刺激期间的差异
stim_diff = np.abs(stim_signal_clean[4:] - stim_signal_clean[:-4])
stim_max_val = np.max(stim_diff)
stim_max_idx = np.argmax(stim_diff) + 4

# 计算clean信号基线期间的差异
base_diff = np.abs(baseline_signal_clean[4:] - baseline_signal_clean[:-4])
base_max_val = np.max(base_diff)
base_max_idx = np.argmax(base_diff) + 4

# 创建保存文件夹（如果不存在）
save_dir = os.path.join(figureRoot, filename.split('.')[0], f'chan{chanNum}')
os.makedirs(save_dir, exist_ok=True)

# 图1：完整信号对比图
plt.figure(figsize=(15, 6))
plt.plot(full_time, full_signal, 'k-', label='Original Signal', alpha=0.5)
plt.plot(full_time, full_signal_clean, 'b-', label='Cleaned Signal', alpha=0.5)

# 添加刺激期间和基线期间的虚线边界
# plt.axvline(x=0, color=RESPONSE_COLOR, linestyle='--', alpha=0.5, label='Stim Period')
# plt.axvline(x=duration, color=RESPONSE_COLOR, linestyle='--', alpha=0.5)
# plt.axvline(x=baseline_start/sampleRate, color=BASELINE_COLOR, linestyle='--', alpha=0.5, label='Baseline Period')
# plt.axvline(x=full_time[-1], color=BASELINE_COLOR, linestyle='--', alpha=0.5)

# 添加刺激期间的三连值（红色）
if PPSamp != 1:
    if stim_max_idx-PPSamp >= 0 and stim_max_idx+PPSamp < len(stim_diff):
        triplet_indices = [stim_max_idx-PPSamp, stim_max_idx, stim_max_idx+PPSamp]
        triplet_times = [t/sampleRate for t in triplet_indices]
        triplet_values = [stim_signal_clean[i] for i in triplet_indices]
        plt.plot(triplet_times, triplet_values, 'o', color =  RESPONSE_COLOR, label='Response Period Triplet Points')

# 添加基线期间的三连值（绿色）
if PPSamp != 1:
    if base_max_idx-PPSamp >= 0 and base_max_idx+PPSamp < len(base_diff):
        triplet_indices = [base_max_idx-PPSamp, base_max_idx, base_max_idx+PPSamp]
        triplet_times = [(t + baseline_start)/sampleRate for t in triplet_indices]
        triplet_values = [baseline_signal_clean[i] for i in triplet_indices]
        plt.plot(triplet_times, triplet_values, 'o', color = BASELINE_COLOR, label='Baseline Period Triplet Points')

plt.legend()
plt.xlabel('Time (ms)', fontsize=12)
plt.ylabel('Amplitude (V)', fontsize=12)
# plt.grid(True)
# plt.xticks([])
# plt.yticks([])
# plt.box(False)
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig(os.path.join(save_dir, 'full_signal.png'), dpi=300, bbox_inches='tight')
plt.close()

# 图2：刺激期间的clean信号
plt.figure(figsize=(10, 6))
plt.plot(stim_time, stim_signal_clean, 'b-', label='Cleaned Signal', alpha=0.6)
if PPSamp != 1:
    if stim_max_idx-PPSamp >= 0 and stim_max_idx+PPSamp < len(stim_diff):
        triplet_indices = [stim_max_idx-PPSamp, stim_max_idx, stim_max_idx+PPSamp]
        triplet_times = [t/sampleRate for t in triplet_indices]
        triplet_values = [stim_signal_clean[i] for i in triplet_indices]
        plt.plot(triplet_times, triplet_values, 'o', color = RESPONSE_COLOR, label='Triplet Points')
plt.legend()
plt.xlabel('Time (ms)', fontsize=12)
plt.ylabel('Amplitude (V)', fontsize=12)
# plt.grid(True)
# plt.xticks([])
# plt.yticks([])
# plt.box(False)
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig(os.path.join(save_dir, 'stim_period.png'), dpi=300, bbox_inches='tight')
plt.close()

# 图3：基线期间的clean信号
plt.figure(figsize=(10, 6))

# 计算新的x轴范围
if PPSamp != 1:
    if base_max_idx-PPSamp >= 0 and base_max_idx+PPSamp < len(base_diff):
        # 获取基线期三连值的中心时间点
        center_time = (baseline_start + base_max_idx)/sampleRate
        # 使用与刺激期相同的时间范围
        half_duration = duration/2
        x_min = center_time - half_duration
        x_max = center_time + half_duration
        
        # 根据新的时间范围筛选数据点
        mask = (baseline_time >= x_min) & (baseline_time <= x_max)
        plot_time = baseline_time[mask]
        plot_signal = baseline_signal_clean[mask]
        
        # 绘制信号
        plt.plot(plot_time, plot_signal, 'b-', label='Cleaned Signal', alpha=0.6)
        
        # 添加三连值点
        triplet_indices = [base_max_idx-PPSamp, base_max_idx, base_max_idx+PPSamp]
        triplet_times = [(t + baseline_start)/sampleRate for t in triplet_indices]
        triplet_values = [baseline_signal_clean[i] for i in triplet_indices]
        plt.plot(triplet_times, triplet_values, 'o', color =  BASELINE_COLOR, label='Triplet Points')
        
        # 设置x轴范围
        plt.xlim(x_min, x_max)

plt.legend()
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Amplitude (V)', fontsize=12)
# plt.grid(True)
# plt.xticks([])
# plt.yticks([])
# plt.box(False)
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig(os.path.join(save_dir, 'baseline_period.png'), dpi=300, bbox_inches='tight')
plt.close()

# 打印clean信号的三连值
print(f"Channel {chanNum} - Stimulation triplet values (Clean signal):")
if PPSamp != 1:
    print(f"Times (s): {[t/sampleRate for t in [stim_max_idx-PPSamp, stim_max_idx, stim_max_idx+PPSamp]]}")
    print(f"Values: {[stim_signal_clean[i] for i in [stim_max_idx-PPSamp, stim_max_idx, stim_max_idx+PPSamp]]}")

print(f"\nChannel {chanNum} - Baseline triplet values (Clean signal):")
if PPSamp != 1:
    base_times = [(t + baseline_start)/sampleRate for t in [base_max_idx-PPSamp, base_max_idx, base_max_idx+PPSamp]]
    print(f"Times (s): {base_times}")
    print(f"Values: {[baseline_signal_clean[i] for i in [base_max_idx-PPSamp, base_max_idx, base_max_idx+PPSamp]]}")

#%%
# import seaborn as sns
# # 获取指定通道的基线期和刺激期normalized三连值
# baseline_vals = stimParCLEAN[f'DIV_baseline{chanNum}'].values
# stim_vals = stimParCLEAN[f'ABS_STD_DIV_chan{chanNum}'].values

# # 计算95th percentile
# baseline_95 = np.percentile(baseline_vals, 95)
# baseline_95_2x = 2 * baseline_95  # 计算基线期95th percentile的2倍

# # 基线期直方图
# plt.figure(figsize=(8, 6))
# sns.histplot(data=baseline_vals, bins=30, color='green')
# plt.axvline(x=baseline_95, color='black', linestyle='--', alpha=0.6)

# # 计算基线期y轴范围并留出空间
# y_min, y_max = plt.ylim()
# plt.ylim(y_min - (y_max-y_min)*0.15, y_max)

# # 获取基线期x轴范围
# x_min, x_max = plt.xlim()
# x_offset = (x_max - x_min) * 0.02

# # 添加基线期95th percentile标注
# plt.text(baseline_95 + x_offset, y_min - (y_max-y_min)*0.08, '95th percentile', 
#          color='black', rotation=0, ha='left', va='top', fontsize=12)

# # 去除title, labels, ticks和外框
# plt.title('')
# plt.xlabel('')
# plt.ylabel('')
# # plt.xticks([])
# plt.yticks([])
# plt.box(False)
# plt.savefig(os.path.join(save_dir, 'baseline_hist.png'), dpi=300, bbox_inches='tight')

# # 刺激期直方图
# plt.figure(figsize=(8, 6))
# sns.histplot(data=stim_vals, bins=30, color='tab:red')
# plt.axvline(x=baseline_95_2x, color='black', linestyle='--', alpha=0.6)

# # 计算刺激期y轴范围并留出空间
# y_min, y_max = plt.ylim()
# plt.ylim(y_min - (y_max-y_min)*0.15, y_max)

# # 获取刺激期x轴范围
# x_min, x_max = plt.xlim()
# x_offset = (x_max - x_min) * 0.02

# # 添加刺激期2x baseline 95th percentile标注
# plt.text(baseline_95_2x + x_offset, y_min - (y_max-y_min)*0.08, '2x 95th percentile', 
#          color='black', rotation=0, ha='left', va='top', fontsize=12)

# # 去除title, labels, ticks和外框
# plt.title('')
# plt.xlabel('')
# plt.ylabel('')
# # plt.xticks([])
# plt.yticks([])
# plt.box(False)
# plt.savefig(os.path.join(save_dir, 'stim_hist.png'), dpi=300, bbox_inches='tight')
#%%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 获取指定通道的基线期和刺激期normalized三连值
baseline_vals = stimParCLEAN[f'DIV_baseline{chanNum}'].values
stim_vals = stimParCLEAN[f'ABS_STD_DIV_chan{chanNum}'].values

# 创建新的图形
plt.figure(figsize=(10, 6))

# 绘制基线期直方图
sns.histplot(data=baseline_vals, bins=30, color=BASELINE_COLOR, alpha=BASELINE_ALPHA, 
            label='Baseline Period', stat='count')

# 绘制刺激期直方图
sns.histplot(data=stim_vals, bins=30, color=RESPONSE_COLOR, alpha=RESPONSE_ALPHA, 
            label='Response Period', stat='count')

# 计算基线期的99百分位数
baseline_99 = np.percentile(baseline_vals, 99)

# 添加99百分位数垂直线
plt.axvline(x=baseline_99, color='black', linestyle='--', alpha=0.8)

# 获取y轴范围并留出空间添加标注
y_min, y_max = plt.ylim()
plt.ylim(y_min - (y_max-y_min)*0.15, y_max)

# 获取x轴范围用于文本标注位置计算
x_min, x_max = plt.xlim()
x_offset = (x_max - x_min) * 0.02

# 添加99百分位数标注
plt.text(baseline_99 + x_offset, y_min - (y_max-y_min)*0.08, 
         '99th percentile', color='black', rotation=0, 
         ha='left', va='top', fontsize=12)

# 添加图例
plt.legend(frameon=False)

# 设置轴标签
plt.xlabel('Normalized triplet value', fontsize=12)
plt.ylabel('Count', fontsize=12)

# 美化图形
plt.title('')
# 设置只显示左边和下边的边框
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

# 保存图形
plt.savefig(os.path.join(save_dir, 'combined_hist.png'), 
            dpi=300, bbox_inches='tight')
plt.close()

# 打印统计信息
print(f"通道 {chanNum} 的统计信息：")
print(f"基线期99百分位数：{baseline_99:.3f}")
print(f"基线期平均值：{np.mean(baseline_vals):.3f}")
print(f"刺激期平均值：{np.mean(stim_vals):.3f}")
print(f"超过99百分位数的刺激期样本比例：{np.mean(stim_vals > baseline_99):.2%}")