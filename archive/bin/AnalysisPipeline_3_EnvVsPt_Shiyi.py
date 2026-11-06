#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 20:48:05 2025

@author: tim
"""

# AnalysisPipeline_3 for ENV vs pt
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
path.append('/home/colliculus/behaviourBoxes/software/ratCageProgramsV3/')
import RZ2ephys as ep
from tqdm import tqdm
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from sys import path
path.append('/home/colliculus/behaviourBoxes/software/ratCageProgramsV3/')


'''
################################################################################################################################
This code is a individual code to analyse the AMUA results of the recorded cue trading data (Step 3)
################################################################################################################################
'''


# %%
'''
write down the important conditions before starting the pipeline- so adjust these parameters and tell the program which data is going to be analyzed. Call this variable: "Data_conds"
makes life maybe easier. 
Cue Trading: [('clickRate (Hz)',1, 100, 900]),[ ('ITD (ms)',[-120, -80, -40, 0, 40, 80, 120]), ('ILD',[-4, -1, 0, 1, 4])]
Env Vs Pt:  [('duration (s)',[0.01, 0.05, 0.2]), ('clickRate (Hz)',[900, 4500]), ('ITD (ms)',[-0.1, 0, 0.1]), ('env ITD (ms)',[-0.1, 0, 0.1])]
Weighting: ?



'''

# specify the stim params that were permuted to generate different stimulus conditions
stimVarParams = ['duration (s)', 'clickRate (Hz)', 'ITD (ms)', 'env ITD (ms)']


# %%

dataRoot = '/ephys/2/0_ENVvsPT/Data'
analysisRoot = '/ephys/2/0_ENVvsPT/Analysis'
figureRoot = '/ephys/2/0_ENVvsPT/Figure'

# %%
'''
Read in & write in
'''


def readEphysFileFromRow(dataset, row, loadOrig=False, loadAMUA=False, loadClean=False):
    if loadOrig == True:
        filename = dataset.fileName[row]
    if loadAMUA == True:
        filename = dataset.fileName[row][:-6]+'_AMUA'
    if loadClean == True:
        filename = dataset.fileName[row][:-6]+'_Clean'
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
            print(filename)
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

# def checkSwpsSize(swps, stimPar, stimVarParams):
#     # for HK data, some set recorded by 64 channel electrode containing IC and AC, only take first 32ch(IC)
#     # for FR data, some set last swp is missed or shorter than normal, replace by second last swp
#     if swps[1].signal.shape[1] != 32:
#         for swp in swps:
#             swp.signal = swp.signal[:, :32]
#     if len(swps) == len(stimPar):
#         # If last swp not recorded
#         Last_conds = findLastCond(stimPar, stimVarParams)
#         Last_swpIdx=findSweepsFor(Last_conds, stimPar)
#         swps = np.append(swps, swps[Last_swpIdx[-2]])

#     # Code by Tim to get rid of the fact that sometimes the one sweep was too short (2* stim duration in sec):
#     # Can also adjust the duration sweeps has to have
#     Short_swps = [] #List of short swps, adjusted by the for loop below...
#     for idx in  range (len(stimPar)):
#         if swps[idx+1].signal.shape[0]/swps[-1].sampleRate < 2*(float(stimPar.iloc[idx]['duration (s)'])):
#             Short_swps.append(idx)
#             print('sweeps too short:', idx)

#     for idx, Short_swp in enumerate (Short_swps):
#         # print(Short_swp)
#         Short_conds = findConditionsForswp(Short_swp, stimPar, stimVarParams)
#         # print(Short_conds)
#         Last_swpIdxs = findSweepsFor(Short_conds, stimPar)
#         for i, Last_swpIdx in enumerate (Last_swpIdxs):
#             if Last_swpIdx == Short_swp+1:
#                 try:
#                     swps[Last_swpIdxs[i]] = swps[Last_swpIdxs[i - 1]]
#                 except:
#                     swps[Last_swpIdxs[i]] = swps[Last_swpIdxs[i + 1]]
#     return swps, stimPar


# def findLastCond(stimPar, stimVarParams):
#     aStim=stimPar.iloc[len(stimPar)-1]
#     Last_conds=[]
#     for aCond in stimVarParams:
#         Last_conds.append( (aCond, f'{aStim[aCond]}') )
#     return Last_conds
# %%
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

    # duration = float(stimPar.iloc[swp-1]['duration (s)'])
    # clickRate = int(stimPar.iloc[swp-1]['clickRate (Hz)'])
    # ITD = float(stimPar.iloc[swp-1]['ITD (ms)'])
    # Env = float(stimPar.iloc[swp-1]['env ITD (ms)'])
    # conds = [('duration (s)',duration), ('clickRate (Hz)',clickRate), ('ITD (ms)',ITD), ('env ITD (ms)',Env)]
    # return (conds)


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


# %%
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


# %%
'''
variance explained ratio calculation     
'''


def calcVarianceExplained(df):
    ss_t = ssTotalCalc(df)
    model = ols('AMUA ~ C(PT_ITD)+C(ENV_ITD)+C(PT_ITD):C(ENV_ITD)',
                data=df).fit()
    result = sm.stats.anova_lm(model, type=1)
    ss_explainedTotal = result.iloc[0, 1]+result.iloc[1, 1]
    ss_pt = result.iloc[0, 1]/ss_explainedTotal
    ss_env = result.iloc[1, 1]/ss_explainedTotal
    # ss_pt = result.iloc[0, 1]/ss_t
    # ss_env = result.iloc[1, 1]/ss_t
    return ss_pt, ss_env, result


def ssTotalCalc(df):
    totalMean = df['AMUA'].mean()
    ss_t = sum((df['AMUA']-totalMean)**2)
    return ss_t


def permutationTestfor(df, num=1000):
    ss_pt_array = []
    ss_env_array = []
    print('start permuting...')
    for tt in range(num):
        shuffled_index = np.random.permutation(df.index)
        df['AMUA'] = df['AMUA'].iloc[shuffled_index].values
        ss_pt, ss_env = calcVarianceExplained(df)
        ss_pt_array.append(ss_pt)
        ss_env_array.append(ss_env)
    ss_pt_p = np.percentile(np.array(ss_pt_array), 95)
    ss_env_p = np.percentile(np.array(ss_env_array), 95)
    return ss_pt_p, ss_env_p


# %%
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


def plot32Channel(swps, conditions, allRepeats=False, avgRepeats=False, ncols=8, nrows=4):
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
    gs.update(left=0.05, right=0.98, bottom=0.12,
              top=0.92, wspace=0.1, hspace=0.2)
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


def plot3X3plot(swps, sub_conditions, chanNum, itd=[-0.1, 0, 0.1], allRepeats=False, avgRepeats=False,):
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
        ax.set_title(rate+'pps_'+dur+'s')
        axs.append(ax)
    return fig, axs


# %%
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
    print(x.shape)
    ax.scatter(x*100, y*100, color=color, facecolors='none', hatch='///')
    ax.set_xlim(-10, 110)
    ax.set_ylim(-10, 110)


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


def plot2X3Scatter(sub_recordingSites, fig, axs, text, durs=[0.01, 0.05, 0.2], pps=[900, 4500], colors=None, type=None):
    # fig, gs_scatter = prepare2X3Plot()
    if type == 'significancy':
        colors = ['tab:blue', 'tab:red', 'purple', 'grey']
        Label = [[1, 0], [0, 1], [1, 1], [0, 0]]
        for ii in range(6):  # loop 2 clickrate*3duration
            ax = axs[ii]
            VEheader, significanceHeader = Header(sub_recordingSites, ii)
            for jj in range(4):
                df = sub_recordingSites[(sub_recordingSites[significanceHeader[0]] == Label[jj][0]) & (
                    sub_recordingSites[significanceHeader[1]] == Label[jj][1])]
                xy = df[VEheader]
                # print(len(xy))
                plotScatter(xy, VEheader, ax, text, color=colors[jj])
    if type == 'penetration':
        for ii in range(6):  # loop 2 clickrate*3duration
            ax = axs[ii]
            VEheader, significanceHeader = Header(sub_recordingSites, ii)
            xy = sub_recordingSites[VEheader]
            plotScatter(xy, VEheader, ax, text, color=colors)
            if ii == 0:
                line = Line2D([0], [0], marker='o', color='w',
                              markerfacecolor='none', markeredgecolor=colors,
                              label=text, markersize=10, markeredgewidth=1.5)
                fig.legend_elements.append(line)
                # Place legend outside all subplots
                # fig.legend(handles=fig.legend_elements,
                #          loc='center left',
                #          bbox_to_anchor=(1.05, 0.5),
                #          fontsize=10)

    return fig


# %%
# Function to check which sweeps have left over artefact:
'''
Previously used in the VE calculation to exclude the data with leftover artefact and exclude if more than 1/4 of the sweeps with the same conds are kicked out
'''

# Old function to find the leftover artefact based on the reject artefact csv file:
# def LeftOverArtefact (filename, channel, rejected_art, stimPar):
#     leftoverart = rejected_art[filename].dropna()
#     if len(leftoverart) == 0:
#         remove_keys = np.empty(0, dtype=object)
#         return remove_keys
#     if len(leftoverart) > 0:
#         leftoverart_chan = leftoverart[leftoverart.str.contains(f"Chan: {channel}", na=False)]

#         sweeps_art = [] # holds all the sweeps for this channel with residual artefact
#         for entry in leftoverart_chan:
#             sweep_art = int(re.search(r"Sweep: (\d+)", entry).group(1))
#             sweeps_art.append(sweep_art)


#         Artefact_check = []     #Holds combinations with too much artefact
#         nup = ep.nonUniqueParams(stimPar)
#         p = ep.permutations(nup, reverse = True)
#         for ind, stim in enumerate(p):
#             idxs_prev = ep.sweepIndecesFor(stimPar, list(nup.keys()), stim)

#          #check if there is residual artefact:
#             idxs = np.array([item for item in idxs_prev if item+1 not in sweeps_art])
#          #Check if left over sweeps are at least 2/3 of the trials presented.
#             if len(idxs) < 0.75 * len(idxs_prev):
#                 # print('Too much leftover aretefact in combination: ', stim)
#                 Artefact_check.append(stim)

#      #After the loop below: holds all the values fo keys that need to be removed:
#         remove_keys = np.empty(len(Artefact_check), dtype=object)
#         for i in range(len(Artefact_check)):
#             remove_keys[i] = ('Dur'+str(Artefact_check[i][0])+'_Rate'+str(Artefact_check[i][1]))
#         remove_keys = np.unique(remove_keys)
#         # print('Too much leftover aretefact in combination: ', remove_keys)
#         return remove_keys

# Old function to find the leftover artefact based on the reject artefact csv file:
# def remove_single_ARTsweeps(idxs_prev, filename, channel, rejected_art):
#     leftoverart = rejected_art[filename].dropna()
#     leftoverart_chan = leftoverart[leftoverart.str.contains(f"Chan: {channel}", na=False)]

#     sweeps_art = []
#     for entry in leftoverart_chan:
#         sweep_art = int(re.search(r"Sweep: (\d+)", entry).group(1))
#         sweeps_art.append(sweep_art)

#     idxs = [item for item in idxs_prev if item+1 not in sweeps_art]
#     idxs = np.array(idxs, dtype=int)

#    return idxs


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
            idxs_prev = ep.sweepIndecesFor(
                stimParCLEAN, list(nup.keys()), stim)
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
            remove_keys[i] = ('Dur'+str(Artefact_check[i][0]) +
                              '_Rate'+str(Artefact_check[i][1]))
        remove_keys = np.unique(remove_keys)
#         # print('Too much leftover aretefact in combination: ', remove_keys)
    return remove_keys


def remove_single_ARTsweeps(idxs_prev, channel, stimParCLEAN):
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

    idxs = [item for item in idxs_prev if item not in leftoverart]
    idxs = np.array(idxs, dtype=int)

    return idxs


# %% VE calc
'''
variance explained calculation
'''
# read in datafileInventory.csv and recordingsiteInventory.csv
dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_V3.csv')
recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventoryV4.csv')
# rejected_art = pd.read_csv(analysisRoot+'/LeftoverArtefact.csv')
#%%
dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_V3.csv')

corhort = 'ND'
animNum = 6
penNum = 3
chanNum = 8
duration = 0.01
clickRate = 900
anim = dataFiles[(dataFiles['condition'] == corhort)
                 ]['animDir'].unique()[animNum]
filename = dataFiles[(dataFiles['animDir'] == anim) & (
    dataFiles['Cleaned'] == 1)].iloc[penNum]['fileName']
print(filename)
# %%
for row in range(len(dataFiles)):
    # if dataFiles.loc[row, 'condition'] in ['ITD+']:
    # if row >= 59:
    if dataFiles.loc[row, 'fileName'] is filename:
    # if row in range(59, 60):
    # if True:
        if dataFiles.Cleaned[row] == 0:
            continue
        fname = dataFiles.fileName[row]
        csvName = fname[:-6]+'_Clean.csv'
        animDir = dataFiles.animDir[row]
        swpAMUA, stimParAMUA = readEphysFileFromRow(dataFiles, row, loadAMUA = True)
        stimParCLEAN = pd.read_csv(dataRoot+'/'+animDir+'/'+csvName)
        # swpCLEAN, stimParCLEAN = readEphysFileFromRow(dataFiles, row, loadClean = True)

        # from datafilename find corresponding recording sites in recordingsiteInventory.csv
        sub_recordingSites = findRecordingSite(dataFiles, row)

        #Check the sweeps with leftover artefact:
        filename = dataFiles.iloc[row]['fileName']
        nup = ep.nonUniqueParams(stimParAMUA, stimVarParams)
        p = ep.permutations(nup, reverse = True)

        # #get only specific dur for Freiburg data to check --> locally saved
        #     #################################################################
        # dur_check = np.any(p == 0.2, axis=1)
        # p = p[dur_check]


        # data selection use new method(2*95% tile)

        # loop over sub_recordingSites, if not, fill blank
        for channel in sub_recordingSites.Channel:
            # if True:
            if channel == chanNum:
            # if channel == 0 or channel == 1:
                idx = sub_recordingSites.index[channel]
                if sub_recordingSites.Responsive[idx]== 1:
                    print ('Now analysing:', filename, 'channel:', channel)

                    remove_keys = LeftOverArtefact (nup, p, channel, stimParCLEAN)   #check here if some keys have to be removed -> keys in that case are durs and clickrates
                    idx = sub_recordingSites.index[channel]

                    CHECKING = []
                    exclude= None
                    for ii, stimulus in enumerate(p):
                        print (p[ii])
                        keyName = 'Dur'+str(p[ii, 0])+'_Rate'+str(p[ii, 1])
                        print(keyName)
                        if keyName in remove_keys:                           #if the key is removed it is not taken here.
                            print('remove')
                            continue

                        idxs_prev = ep.sweepIndecesFor(stimParCLEAN, list(nup.keys()), stimulus)
                        #from 0 to 1079


                        #check if there is residual artefact and remove single trials if less more than 1/4 still good:
                        idxs = remove_single_ARTsweeps (idxs_prev, channel, stimParCLEAN)
                        CHECKING.append(idxs+1)

                        signal = poolChannel(swpAMUA[idxs+1], channel)[:int(stimulus[0]*swpAMUA[0].sampleRate), :].mean(axis = 0) #Warning: idxs = idxs -1 originally due to ep.sweepIndecesFor

                        if ii % 9 == 0:
                            df = pd.DataFrame({'PT_ITD': stimulus[2].repeat(len(signal)), 'ENV_ITD': stimulus[3].repeat(len(signal)), 'AMUA': signal})
                        else:
                            df = pd.concat([df,pd.DataFrame({'PT_ITD': stimulus[2].repeat(len(signal)), 'ENV_ITD': stimulus[3].repeat(len(signal)), 'AMUA': signal})], ignore_index=True)
                        if ii % 9 == 8 and exclude == None:
                            print(len(df))
                            # ss_pt, ss_env = calcVarianceExplained(df)
                            ss_pt, ss_env, result = calcVarianceExplained(df)
                            # ss_pt_p, ss_env_p = permutationTestfor(df, 1000)

                            # ss_pt_p = 0
                            # ss_env_p = 0

                            # recordingSites = writeValueinto(recordingSites, idx, ss_pt, ss_env, ss_pt_p, ss_env_p, keyName)
# print(recordingSites.iloc[idx])
# recordingSites.to_csv(analysisRoot+'/'+'recordingsiteInventory_EVE.csv', index=False)
#%%
plt.figure(figsize=(10, 6))
for pt_itd in df['PT_ITD'].unique():
    subset = df[df['PT_ITD'] == pt_itd]['AMUA']
    plt.hist(subset, alpha=0.5, label=f'PT_ITD={pt_itd}', density=True, bins = 20)
plt.legend()
plt.title('Distribution of AMUA for different PT_ITD values')
plt.xlabel('AMUA')
plt.ylabel('Density')
plt.show()

plt.figure(figsize=(10, 6))
for pt_itd in df['ENV_ITD'].unique():
    subset = df[df['ENV_ITD'] == pt_itd]['AMUA']
    plt.hist(subset, alpha=0.5, label=f'ENV_ITD={pt_itd}', density=True, bins = 20)
plt.legend()
plt.title('Distribution of AMUA for different ENV_ITD values')
plt.xlabel('AMUA')
plt.ylabel('Density')
plt.show()
#%% concatenate FB data into EVE.csv
recordingSites = pd.read_csv(analysisRoot+'/'+'recordingsiteInventory_EVE.csv')
FB = pd.read_csv(analysisRoot+'/'+'recordingsiteInventory_EVE_Freiburg.csv')
l = len(recordingSites)
idx1 = FB[FB['condition'] == 'ITD+'].index[0]
df2 = FB[idx1:]
df1 = recordingSites[:idx1]
df = pd.concat([df1, df2], axis = 0)
df.to_csv(analysisRoot+'/'+'recordingsiteInventory_EVE.csv', index=False)
#%%
'''
plot the left over sweeps that are used for VE calculation since we check for leftover artefacts:
'''


def plot3X3plot_Sanity(swps, sub_conditions, filename, rejected_art, chanNum, itd=[-0.1, 0, 0.1], allRepeats=False, avgRepeats=False,):
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
        idxs_prev = findSweepsFor(conditions.copy(), stimPar)
        leftoverart = rejected_art[filename].dropna()
        leftoverart_chan = leftoverart[leftoverart.str.contains(
            f"Chan: {chanNum}", na=False)]
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
            signal = signal.mean(axis=1)[:len(t)]
        ax.plot(t, signal)
        ax.axvline(x=duration, color='k', linestyle='--')
    return fig


def plot32Channel_Sanity(swps, conditions, allRepeats=False, avgRepeats=False, ncols=8, nrows=4):
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


# CHECK_LEN = []
# for row in range(len(dataFiles)):
#     # row = 61
#     # if dataFiles.iloc[row]['Cleaned'] == 0:
#     #     continue
#     fname = dataFiles.fileName[row]
#     csvName = fname[:-6]+'_Clean.csv'
#     animDir = dataFiles.animDir[row]
#     stimParCLEAN = pd.read_csv(dataRoot+'/'+animDir+'/'+csvName)
#     # swpCLEAN, stimParCLEAN = readEphysFileFromRow(dataFiles, row, loadClean = True)
#     # from datafilename find corresponding recording sites in recordingsiteInventory.csv
#     sub_recordingSites = findRecordingSite(dataFiles, row)

#     #Check the sweeps with leftover artefact:
#     filename = dataFiles.iloc[row]['fileName']

#     # data selection use new method(2*95% tile)

#     # loop over sub_recordingSites, if not, fill blank
#     for channel in sub_recordingSites.Channel:

#         print ('Now analysing:', filename, 'channel:', channel)

#         Headername = f'ABS_STD_DIV_chan{channel}'  # Ensure exact match
#         abs_std_columns = [col for col in stimParCLEAN.columns if col == Headername]

#         # Create a new DataFrame with only ABS_STD columns
#         df_abs_std = stimParCLEAN[abs_std_columns]
#         max_val = np.max(df_abs_std)
#         CHECK_LEN.append(max_val)

# %%
'''
VE distribution plot
'''
# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_not_clean_check.csv')
dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_V3.csv')
# recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventoryV2_not_clean_check.csv')
# recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventory_EVE.csv')
# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory.csv')
recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventoryV2.csv')
corhort_list = ['AD + CI acute', 'ND + CI acute', 'ND + CI experience', 'Burst', 'ITD-']
corhorts = dataFiles.condition.unique()
for corhNum, corhort in enumerate(corhorts):
    if True:
    # if corhNum == 2:
        fig, axs = prepare2X3Plot()
        anims = dataFiles[dataFiles['condition']
                          == corhort]['animDir'].unique()
        print(anims)
        unitNum = 0
        for animNum, anim in enumerate(anims):
            filenames = dataFiles[(dataFiles['animDir'] == anim) & (
                dataFiles['Cleaned'] == 1)]['fileName']
            for penNum, filename in enumerate(filenames):
                sub_recordingSites = recordingSites[(
                    recordingSites['fileName'] == filename) & (recordingSites['Responsive'] == 1)]
                unitNum += len(sub_recordingSites)
                scatterText = 'anim'+str(animNum)+'pen'+str(penNum)
                fig = plot2X3Scatter(
                    sub_recordingSites, fig, axs, scatterText, type='significancy')
            fig.text(0, 0.5, 'Variance explained by PT_ITD (%)',
                     va='center', rotation=-90, fontsize=14)
            fig.text(0.38, 0.053, 'Variance explained by ENV_ITD (%)',
                     va='center', fontsize=14)
            # fig.suptitle(corhort+'_ResponsiveUnit#'+str(unitNum), fontsize=14)
            fig.suptitle(corhort_list[corhNum]+'_ResponsiveUnit#'+str(unitNum)+'_EVE', fontsize=14)
            # plt.savefig('/ephys/2/0_ENVvsPT/Figure/Result_ScatterPlot/'+corhort
#%%
'''
VE distribution plot, penetration label by different color
'''
# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_not_clean_check.csv')
dataFiles = pd.read_csv(analysisRoot+'/datafileInventory.csv')
# recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventoryV2_not_clean_check.csv')
recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventoryV2.csv')
# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory.csv')
# recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventoryV2.csv')

corhorts = dataFiles.condition.unique()
colors = plt.cm.tab20(np.random.rand(40))
for corhNum, corhort in enumerate(corhorts):
    if corhNum == 1:
        # if True:
        fig, axs = prepare2X3Plot()
        if not hasattr(fig, 'legend_elements'):
            fig.legend_elements = []
        anims = dataFiles[dataFiles['condition']
                          == corhort]['animDir'].unique()
        print(anims)
        unitNum = 0
        coloridx = 0
        for animNum, anim in enumerate(anims):
            filenames = dataFiles[(dataFiles['animDir'] == anim) & (
                dataFiles['Cleaned'] == 1)]['fileName']
            for penNum, filename in enumerate(filenames):
                sub_recordingSites = recordingSites[(
                    recordingSites['fileName'] == filename) & (recordingSites['Responsive'] == 1)]
                unitNum += len(sub_recordingSites)
                scatterText = 'anim'+str(animNum)+'pen'+str(penNum)
                # fig = plot2X3Scatter(sub_recordingSites, fig, axs, scatterText, type = 'significancy')
                coloridx += 1
                color = colors[coloridx]
                fig = plot2X3Scatter(
                    sub_recordingSites, fig, axs, scatterText, type='penetration', colors=color)
                print(fig.legend_elements)
            fig.legend(handles=fig.legend_elements,
                       loc='center left',
                       bbox_to_anchor=(1.05, 0.5),
                       fontsize=10)
            fig.text(0, 0.5, 'Variance explained by PT_ITD (%)',
                     va='center', rotation=-90, fontsize=14)
            fig.text(0.38, 0.053, 'Variance explained by ENV_ITD (%)',
                     va='center', fontsize=14)
        fig.suptitle(corhort+'_ResponsiveUnit#'+str(unitNum), fontsize=14)
        leg = fig.legend(handles=fig.legend_elements,
                         bbox_to_anchor=(0.98, 0.98),
                         fontsize=10)
        leg.set_zorder(1000)
        # plt.savefig('/ephys/2/0_ENVvsPT/Figure/Result_ScatterPlot/'+corhort
# %%
colors = [
    '#FF0000',
    '#00FF00',
    '#0000FF',
    '#FFD700',
    '#FF00FF',
    '#00FFFF',
    '#FF8C00',
    '#8B008B',
    '#4682B4',
    '#32CD32'
]
# %%
'''
VE distribution plot, for signal animal
'''
# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_not_clean_check.csv')
dataFiles = pd.read_csv(analysisRoot+'/datafileInventory.csv')
# recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventoryV2_not_clean_check.csv')
recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventoryV2.csv')
# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory.csv')
# recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventoryV2.csv')

corhorts = dataFiles.condition.unique()
# colors = plt.cm.tab20(np.linspace(0, 1, 40))
for corhNum, corhort in enumerate(corhorts):
    if corhNum == 3:
        # if True:
        # fig, axs = prepare2X3Plot()
        # if not hasattr(fig, 'legend_elements'):
        #     fig.legend_elements = []
        anims = dataFiles[dataFiles['condition']
                          == corhort]['animDir'].unique()
        print(anims)
        unitNum = 0
        # coloridx = 0
        for animNum, anim in enumerate(anims):
            if animNum == 2:
                # if True:
                coloridx = 0
                fig, axs = prepare2X3Plot()
                if not hasattr(fig, 'legend_elements'):
                    fig.legend_elements = []
                filenames = dataFiles[(dataFiles['animDir'] == anim) & (
                    dataFiles['Cleaned'] == 1)]['fileName']
                for penNum, filename in enumerate(filenames):
                    if True:
                        # if penNum in [0, 1]:
                        sub_recordingSites = recordingSites[(
                            recordingSites['fileName'] == filename) & (recordingSites['Responsive'] == 1)]
                        unitNum += len(sub_recordingSites)
                        scatterText = 'anim'+str(animNum)+'pen'+str(penNum)
                        # fig = plot2X3Scatter(sub_recordingSites, fig, axs, scatterText, type = 'significancy')
                        coloridx += 1
                        color = colors[coloridx]
                        fig = plot2X3Scatter(
                            sub_recordingSites, fig, axs, scatterText, type='penetration', colors=color)
                    fig.legend(handles=fig.legend_elements,
                               loc='center left',
                               bbox_to_anchor=(1.05, 0.5),
                               fontsize=10)
                fig.text(0, 0.5, 'Variance explained by PT_ITD (%)',
                         va='center', rotation=-90, fontsize=14)
                fig.text(0.38, 0.053, 'Variance explained by ENV_ITD (%)',
                         va='center', fontsize=14)
                fig.suptitle(corhort+' '+anim, fontsize=14)
                leg = fig.legend(handles=fig.legend_elements,
                                 bbox_to_anchor=(0.98, 0.98),
                                 fontsize=10)
                leg.set_zorder(1000)
                # plt.savefig('/ephys/2/0_ENVvsPT/Figure/Result_ScatterPlot/'+corhort+anim)
                # plt.close(fig)
# %%
'''
plot out all channel(chs) which the VEpt exceed setting range
'''
checkParams = [4500.0, 0.2]

# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_not_clean_check.csv')
dataFiles = pd.read_csv(analysisRoot+'/datafileInventory.csv')
# recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventoryV2_not_clean_check.csv')
recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventoryV2.csv')
# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory.csv')
# recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventoryV2.csv')

corhorts = dataFiles.condition.unique()
# colors = plt.cm.tab20(np.linspace(0, 1, 40))
for corhNum, corhort in enumerate(corhorts):
    if corhNum == 0:
        # if True:
        # fig, axs = prepare2X3Plot()
        # if not hasattr(fig, 'legend_elements'):
        #     fig.legend_elements = []
        anims = dataFiles[dataFiles['condition']
                          == corhort]['animDir'].unique()
        print(anims)
        unitNum = 0
        # coloridx = 0
        for animNum, anim in enumerate(anims):
            if animNum == 5:
                # if True:
                # coloridx = 0
                # fig, axs = prepare2X3Plot()
                # if not hasattr(fig, 'legend_elements'):
                # fig.legend_elements = []
                filenames = dataFiles[(dataFiles['animDir'] == anim) & (
                    dataFiles['Cleaned'] == 1)]['fileName']
                for penNum, filename in enumerate(filenames):
                    if penNum == 1:
                        # if True:
                        # if penNum in [0, 1]:
                        sub_recordingSites = recordingSites[(
                            recordingSites['fileName'] == filename) & (recordingSites['Responsive'] == 1)]
                        key = 'Dur' + \
                            str(checkParams[1])+'_Rate' + \
                            str(checkParams[0])+'_VEpt'
                        i = np.where(sub_recordingSites[key] > 0.8)[0]
                        targetidx = sub_recordingSites.index[i]
                        target = sub_recordingSites.iloc[i]
                        chs = target.Channel
# %%
'''
debug VE distribution
'''
dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_V3.csv')

corhort = 'ND'
animNum = 6
penNum = 3
chanNum = 8
duration = 0.01
clickRate = 900

anim = dataFiles[(dataFiles['condition'] == corhort)]['animDir'].unique()[animNum]
filename = dataFiles[(dataFiles['animDir'] == anim) & (dataFiles['Cleaned'] == 1)].iloc[penNum]['fileName']
print(filename)
swpsClean, stimParCLEAN = ep.readEphysFile(
    dataRoot+'/'+anim+'/'+filename[:-6]+'_Clean')
swpsAMUA, stimPar = ep.readEphysFile(
    dataRoot+'/'+anim+'/'+filename[:-6]+'_AMUA')

# %% plot clean data
# for chanNum in chs:
if True:
    sampleRate = swpsClean[1].sampleRate
    t = np.arange(0, duration+0.05, 1/sampleRate)
    fig, axs = prepare3X3Plot()

    itd = [-0.1, 0, 0.1]
    for ii in range(9):
        xx = ii % 3
        yy = ii//3
        pt_itd = itd[yy]
        env_itd = itd[xx]
        ax = axs[ii]
        print(ax.title, pt_itd, env_itd)
        # conditions=[('duration (s)',duration), ('clickRate (Hz)',clickRate), ('ITD (ms)',pt_itd), ('env ITD (ms)',env_itd)]
        # idxs_prev =findSweepsFor(conditions.copy(), stimPar)
        keys = ['duration (s)', 'clickRate (Hz)', 'ITD (ms)', 'env ITD (ms)']
        conds = np.array([duration, clickRate, pt_itd, env_itd])
        idxs_prev = ep.sweepIndecesFor(stimParCLEAN, keys, conds)

        # idxs_prev = idxs_prev + 1
        # signal = poolChannel(swpsClean[idxs_prev], chanNum)

        swpIdx = remove_single_ARTsweeps(idxs_prev, chanNum, stimParCLEAN)
        swpIdx = swpIdx + 1
        signal = poolChannel(swpsClean[swpIdx], chanNum)

        signal = (signal-signal[0, :])[:len(t)]
        # signal = (signal-signal[0, :])[:len(t)].mean(axis = 1)
        ax.plot(t, signal)
        ax.axvline(x=duration, color='k', linestyle='--')
        # ax.set_title(chanNum)
#%% 
for row in range(len(dataFiles)):
    # if dataFiles.loc[row, 'condition'] in ['ITD+']:
    # if row >= 59:
    # if row in range(59, 60):
    # if dataFiles['fileName'] == filename
    # if True:
        if dataFiles.Cleaned[row] == 0:
            continue
        fname = dataFiles.fileName[row]
        csvName = fname[:-6]+'_Clean.csv'
        animDir = dataFiles.animDir[row]
        swpAMUA, stimParAMUA = readEphysFileFromRow(dataFiles, row, loadAMUA = True)
        stimParCLEAN = pd.read_csv(dataRoot+'/'+animDir+'/'+csvName)
        # swpCLEAN, stimParCLEAN = readEphysFileFromRow(dataFiles, row, loadClean = True)

        # from datafilename find corresponding recording sites in recordingsiteInventory.csv
        sub_recordingSites = findRecordingSite(dataFiles, row)

        #Check the sweeps with leftover artefact:
        filename = dataFiles.iloc[row]['fileName']
        nup = ep.nonUniqueParams(stimParAMUA, stimVarParams)
        p = ep.permutations(nup, reverse = True)

        # #get only specific dur for Freiburg data to check --> locally saved
        #     #################################################################
        # dur_check = np.any(p == 0.2, axis=1)
        # p = p[dur_check]


        # data selection use new method(2*95% tile)

        # loop over sub_recordingSites, if not, fill blank
        for channel in sub_recordingSites.Channel:
            if True:
            # if channel == 0 or channel == 1:
                idx = sub_recordingSites.index[channel]
                if sub_recordingSites.Responsive[idx]== 1:
                    print ('Now analysing:', filename, 'channel:', channel)

                    remove_keys = LeftOverArtefact (nup, p, channel, stimParCLEAN)   #check here if some keys have to be removed -> keys in that case are durs and clickrates
                    idx = sub_recordingSites.index[channel]

                    CHECKING = []
                    exclude= None
                    for ii, stimulus in enumerate(p):
                        print (p[ii])
                        keyName = 'Dur'+str(p[ii, 0])+'_Rate'+str(p[ii, 1])
                        print(keyName)
                        if keyName in remove_keys:                           #if the key is removed it is not taken here.
                            print('remove')
                            continue

                        idxs_prev = ep.sweepIndecesFor(stimParCLEAN, list(nup.keys()), stimulus)
                        #from 0 to 1079


                        #check if there is residual artefact and remove single trials if less more than 1/4 still good:
                        idxs = remove_single_ARTsweeps (idxs_prev, channel, stimParCLEAN)
                        CHECKING.append(idxs+1)

                        signal = poolChannel(swpAMUA[idxs+1], channel)[:int(stimulus[0]*swpAMUA[0].sampleRate), :].mean(axis = 0) #Warning: idxs = idxs -1 originally due to ep.sweepIndecesFor

                        if ii % 9 == 0:
                            df = pd.DataFrame({'PT_ITD': stimulus[2].repeat(len(signal)), 'ENV_ITD': stimulus[3].repeat(len(signal)), 'AMUA': signal})
                        else:
                            df = pd.concat([df,pd.DataFrame({'PT_ITD': stimulus[2].repeat(len(signal)), 'ENV_ITD': stimulus[3].repeat(len(signal)), 'AMUA': signal})], ignore_index=True)
                        if ii % 9 == 8 and exclude == None:
                            print(len(df))
                            ss_pt, ss_env = calcVarianceExplained(df)
                            ss_pt_p, ss_env_p = permutationTestfor(df, 1000)

                            # ss_pt_p = 0
                            # ss_env_p = 0

                            recordingSites = writeValueinto(recordingSites, idx, ss_pt, ss_env, ss_pt_p, ss_env_p, keyName)
print(recordingSites.iloc[idx])
# %%
'''
Correct for missing significance vals in "recordingsiteInvenoryV3"
'''
# import matplotlib.gridspec as gridspec
# from matplotlib.lines import Line2D
# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_V3.csv')
# recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventoryV3.csv')

# corhorts = dataFiles.condition.unique()
# animDir = dataFiles.animDir.unique()
# #Get the headers
# headers = recordingSites.columns
# Data_header = headers[6:]
# significance_headers = [col for col in Data_header if "significance" in col]
# VE_headers = [col for col in Data_header if "significance" not in col and "pValue" not in col]
# pValue_headers = [col for col in Data_header if "pValue" in col]

# for i in range (len(VE_headers)):
#     Norm_head = recordingSites[VE_headers[-2]]
#     p_val_head = recordingSites[pValue_headers[-2]]
#     sig_head = recordingSites[significance_headers[-2]]

#     result = np.where(np.isnan(Norm_head) | np.isnan(p_val_head), np.nan, Norm_head > p_val_head)

# comparison = Norm_head.where(Norm_head.isna() | p_val_head, Norm_head > p_val_head)


# %%
'''
box plots
'''
dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_V3.csv')
recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventoryV3.csv')

corhorts = dataFiles.condition.unique()
corhorts = corhorts[:3]

animDir = dataFiles.animDir.unique()
# Get the headers
headers = recordingSites.columns
Data_header = headers[6:]
significance_headers = [col for col in Data_header if "significance" in col]
VE_headers = [
    col for col in Data_header if "significance" not in col and "pValue" not in col]
pValue_headers = [col for col in Data_header if "pValue" in col]

# jusit a quick and dirty way to get the headers for a plot...
Headers_Plot = []
# This is to get the VE letters --> Jan wated that for the plot
VE_kind = [header.split('_', 2)[-1] for header in VE_headers]
Dur_kind = [header.split('_', 1)[0] for header in VE_headers]
pps_kind = [header.split('_')[1] for header in VE_headers]


for i in (range(len(VE_headers))):
    # print (head)

    Headers_Plot.append(VE_kind[i] + ' ' + 'Dur:' +
                        Dur_kind[i][3:] + ' pps:' + pps_kind[i][4:])

Anims = []
for anim in animDir:
    filenames = dataFiles[dataFiles['animDir'] == anim]['fileName']
    sub_recordingSites = []
    for filename in filenames:
        sub_recordingSites.append(
            recordingSites[recordingSites['fileName'] == filename])
    sub_recordingSites = pd.concat(sub_recordingSites, ignore_index=True)
    sub_recordingSites = sub_recordingSites[sub_recordingSites['Responsive'] == 1]
    Anims.append(sub_recordingSites)


# def create_subplot ():
#     fig, axes = plt.subplots(3, 1)

# For a cohort --> Boxplot all units:
OnlycohortStr = input(
    'Want to check only one cohort[Cohortname/n] type [False] for checking single animal or type [all] for all relevant cohorts (NH, ND and ITD sensitive)?')
if OnlycohortStr != 'n' and OnlycohortStr != 'all':
    Onlycohort = int(OnlycohortStr)

if OnlycohortStr == 'all':
    Onlycohort = None
    corNum = None

Singleanim = None

if OnlycohortStr == 'n':
    Singleanim = input('Want to check a single anim [y/n]?')

    if Singleanim:
        corh = input('Cohort No?')
        Onlycohort = int(corh)
        anim_check = input('Anim No?')
        anim_check = int(anim_check)

if OnlycohortStr == 'all':
    fig_box, axes_box = plt.subplots (3, 1, figsize = (8, 12), sharex = True)
    
    fig_bar, axes_bar = plt.subplots (3, 1, figsize = (8, 12), sharex = True)

    fig_6_bar, axes_6_bar = plt.subplots (2, 3, figsize = (15, 10), sharey = True)

for corhNum, corhort in enumerate(corhorts):
    # if type(Onlycohort) == int :
    # if corhNum != Onlycohort and OnlycohortStr != 'n' and OnlycohortStr != 'all':
    if isinstance (Onlycohort, int ) and corhNum != Onlycohort:
        continue
        # plt.figure()
    # print(corhort)
    all_units = []
    All_data = []
    Sig_data = []
    No_Animals = 0
    # All_labels = []
    for idx, anim in enumerate(Anims):
        # print(idx)
        if idx == 22 or idx == 19:
            continue
        # print (Anims[idx].condition.iloc[0])
        if Anims[idx].condition.iloc[0] == corhort:
            No_Animals += 1
            # print(corhort)
            # Check here how ,many UNITS!
            Units = len(anim)
            all_units.append(Units)
            All_data.append(anim[VE_headers]*100)
            Sig_data.append(anim[significance_headers])
            # All_labels.append(Headers_Plot)

    if Singleanim:
        Combined_Data = All_data[anim_check]
        # Combined_Data = All_data[anim_check].tail(1).index[0] + 1
        Combined_Data = [Combined_Data[col].dropna()
                         for col in Combined_Data.columns]
    else:
        Combined_Data = pd.concat(All_data)
        Combined_Data = [Combined_Data[col].dropna()
                         for col in Combined_Data.columns]

    # Combined_Data = Combined_Data.dropna(axis=1, how='all')
    # Combined_Data = Combined_Data.fillna(1)
    No_All_Units = np.sum(all_units)
    
    #################################### Plot for comparison of single cohort data and single anims data ####################################

    if not OnlycohortStr == 'all':
        plt.figure()
        plt.boxplot(Combined_Data, labels=Headers_Plot)
    
    
        # Code to find out the no. of significant units:
        if Singleanim:
            Combined_Data_Sig = Sig_data[anim_check]
            # Combined_Data = All_data[anim_check].tail(1).index[0] + 1
            Combined_Data_Sig = [Combined_Data_Sig[col].dropna()
                                 for col in Combined_Data_Sig.columns]
        else:
            Combined_Data_Sig = pd.concat(Sig_data)
            Combined_Data_Sig = [Combined_Data_Sig[col].dropna()
                                 for col in Combined_Data_Sig.columns]
        
        
        Percentage_vals = []
        for i_Sig, val_Sig in enumerate(Combined_Data_Sig):
            # print(i_Sig)
            for i, val in enumerate(Combined_Data):
                # print(val)
                if i == i_Sig:
                    n = len(val_Sig)
                    n2 = len(val)
                    Sig = sum(val_Sig)
                    Percentage = round((Sig/n)*100, 2)
                    Percentage_vals.append(Percentage)
                    plt.text(
                        i_Sig + 1, -1, f"Units : {n}", ha='center', fontsize=10, color='blue')
                    # plt.text (i_Sig + 1, -2, f"Units2 : {n2}", ha='center', fontsize=10, color='blue')
                    plt.text(
                        i_Sig + 1, -3, f"% : {Percentage}", ha='center', fontsize=10, color='blue')
    
    
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.axvline(x=2.5, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=4.5, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=6.5, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=8.5, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=10.5, color='black', linestyle='--', linewidth=1)
        
        if corhort == 'ITD+':
            corhort = 'ITD sensitive'
        
        title = f"{corhort}, All units: {No_All_Units} , No. animals: {No_All_Units}"
        plt.title(title)
        
        if Singleanim:
            title = f"{corhort}, All units: {No_All_Units} , No. animal: {anim_check}"
            plt.title(title)
    
        plt.ylabel('VE Values')
        plt.show()
    
        #Barplot from here 
        plt.figure()
        cols = ['green', 'purple'] * (len(Headers_Plot) // 2)  # Create a list with alternating green, purple
        for ind_head, Head in enumerate (Headers_Plot):
            bars = plt.bar(Head, Percentage_vals[ind_head], color = cols[ind_head])
            for bar in bars:
                height = bar.get_height()  # Get bar height
                plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        title = f"{corhort}, All units: {No_All_Units} , No. animals: {No_Animals}"
        plt.title(title)
    
        plt.ylabel('Percentage Significant')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.axvline(x=1.5, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=3.5, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=5.5, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=7.5, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=9.5, color='black', linestyle='--', linewidth=1)
        plt.show()


    #################################### Plot for comparison of all data ####################################
    if OnlycohortStr == 'all':      
        # fig, axes = plt.subplots(3, 1)

    
        
            # Code to find out the no. of significant units:
        if Singleanim:
            Combined_Data_Sig = Sig_data[anim_check]
                # Combined_Data = All_data[anim_check].tail(1).index[0] + 1
            Combined_Data_Sig = [Combined_Data_Sig[col].dropna()
                                    for col in Combined_Data_Sig.columns]
        else:
            Combined_Data_Sig = pd.concat(Sig_data)
            Combined_Data_Sig = [Combined_Data_Sig[col].dropna()
                                    for col in Combined_Data_Sig.columns]
            
            
        Percentage_vals = []
        Units_cond = []
        for i_Sig, val_Sig in enumerate(Combined_Data_Sig):
            # print(i_Sig)
            for i, val in enumerate(Combined_Data):
                # print(val)
                if i == i_Sig:
                    n = len(val_Sig)
                    n2 = len(val)
                    Sig = sum(val_Sig)
                    Percentage = round((Sig/n)*100, 2)
                    Percentage_vals.append(Percentage)
                    Units_cond.append(n)
                    # plt.text(
                    #     i_Sig + 1, -1, f"Units : {n}", ha='center', fontsize=10, color='blue')
                    # # plt.text (i_Sig + 1, -2, f"Units2 : {n2}", ha='center', fontsize=10, color='blue')
                    # plt.text(
                    #     i_Sig + 1, -3, f"% : {Percentage}", ha='center', fontsize=10, color='blue')
        
        
        
        for i_box, ax_box in enumerate (axes_box):
            if i_box == corhNum:
                ax_box.boxplot(Combined_Data, labels=Headers_Plot)
        
                
                ax_box.axvline(x=2.5, color='black', linestyle='--', linewidth=1)
                ax_box.axvline(x=4.5, color='black', linestyle='--', linewidth=1)
                ax_box.axvline(x=6.5, color='black', linestyle='--', linewidth=1)
                ax_box.axvline(x=8.5, color='black', linestyle='--', linewidth=1)
                ax_box.axvline(x=10.5, color='black', linestyle='--', linewidth=1)
                
                if corhort == 'ITD+':
                    corhort = 'ITD sensitive'
                title = f"{corhort}, All units: {No_All_Units} , No. animals: {No_Animals}"
                ax_box.set_title(title)
            
                ax_box.set_ylabel('VE Values', fontsize = 14)
                
            ax_box.tick_params(axis = 'x', rotation=45, labelsize = 10)
            plt.tight_layout()
            plt.show()
    
        # fig, axes = plt.subplots(3, 1)
        
        
        #Barplot with each cohort seperately in a subplot (3,1)
        # plt.figure()
        cols = ['green', 'purple'] * (len(Headers_Plot) // 2)  # Create a list with alternating green, purple
        for i_bar, ax_bar in enumerate (axes_bar):
            if i_bar == corhNum:
                for ind_head, Head in enumerate (Headers_Plot):
                    bars = ax_bar.bar(Head, Percentage_vals[ind_head], color = cols[ind_head])
                    title = f"{corhort}, All units: {No_All_Units} , No. animals: {No_Animals}"
                    ax_bar.set_title(title, fontsize = 17)
                
                    ax_bar.set_ylabel('Percentage Significant', fontsize = 14)
                    ax_bar.axvline(x=1.5, color='black', linestyle='--', linewidth=1)
                    ax_bar.axvline(x=3.5, color='black', linestyle='--', linewidth=1)
                    ax_bar.axvline(x=5.5, color='black', linestyle='--', linewidth=1)
                    ax_bar.axvline(x=7.5, color='black', linestyle='--', linewidth=1)
                    ax_bar.axvline(x=9.5, color='black', linestyle='--', linewidth=1)
                    for bar in bars:
                        height = bar.get_height()  # Get bar height
                        ax_bar.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', 
                                ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            custom_legend = [
                Line2D([0], [0], color='green', lw=8, label='Pt'),  
                Line2D([0], [0], color='purple', lw=8, label='Env') 
                ]
            
            plt.subplots_adjust(top = 0.6)
            plt.legend(handles=custom_legend, loc='upper right', fontsize=12)
            
            ax_bar.tick_params(axis = 'x', rotation=45, labelsize = 10)
            plt.tight_layout()
            plt.show()
    
    
    
        #Barplot with each cohort seperately in a 6er subplot (2,3)
        # plt.figure()
        cols = ['green', 'purple'] * (len(Headers_Plot) // 2)  # Create a list with alternating green, purple
        for i_6_bar, ax_6_bar in enumerate (axes_6_bar.flatten()):
            # print('Loopy', i_6_bar % len(corhorts))
            headers_6_bar = []
            percentage_6_bar = []
            count = 0
            for i_perc, perc in enumerate(Percentage_vals):
                # print(i_perc)
                headers_6_bar.append(Headers_Plot[count: count+2])
                percentage_6_bar.append(Percentage_vals[count: count+2])
                count += 2
                if count >= 12:
                    break
            
        
            modified_headers_6_bar = [f"{s}_{corhNum}" for s in headers_6_bar[i_6_bar]]
            # print('mod_headers', modified_headers_6_bar)
            # print('percentages', Percentage_vals[i_6_bar])
            bars = ax_6_bar.bar(modified_headers_6_bar, percentage_6_bar[i_6_bar])
            for bar in bars:
                height = bar.get_height()  # Get bar height
                ax_6_bar.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
        
            # ax_6_bar.tick_params(axis = 'x', rotation=45, labelsize=10)
            
            ax_6_bar.set_xticklabels(['VEpt', 'VEenv', 'VEpt', 'VEenv', 'VEpt', 'VEenv'], fontsize = 15)
            
            # ax_6_bar.xaxis.set_visible(False)  # Hides everything on the x-axis

            
            if i_6_bar == 0 or i_6_bar == 3:
                ax_6_bar.set_ylabel('Percentage Significant', fontsize=15)
            
            
            plt.tight_layout()
            custom_legend = [
                Line2D([0], [0], color='blue', lw=8, label='NH'),  
                Line2D([0], [0], color='orange', lw=8, label='ND'), 
                Line2D([0], [0], color='green', lw=8, label='ITD sens') 
                ]

            # Add the custom legend to the plot
            if corhNum == 2:
                title = f"Combination: {modified_headers_6_bar[0][5:-2]}"
                ax_6_bar.set_title(title, fontsize = 17)
            plt.subplots_adjust(top = 0.9)
           
            plt.legend(handles=custom_legend, loc='upper right', fontsize=12)
            plt.show()



    
#Adjust the layout such that 900 is above and 4500 below
# axes_6_bar[0, 0].set_position([0, 0])  # Subplot 0 stays as is

# # Subplot (1) becomes (3)
# axes_6_bar[0, 1].set_position([1, 0])  # Subplot 1 moves to position 3

# # Subplot (2) becomes (1)
# axes_6_bar[0, 2].set_position([0, 1])  # Subplot 2 moves to position 1

# # Subplot (4) becomes (3)
# axes_6_bar[1, 1].set_position([0, 2])  

# # Subplot (5) stays as is (position 5)
# axes_6_bar[1, 0].set_position([1, 1]) 

# # Subplot (5) stays as is (position 5)
# axes_6_bar[2, 2].set_position([2, 2]) 

    
#%%


