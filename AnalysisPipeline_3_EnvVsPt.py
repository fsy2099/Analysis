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


'''
write down the important conditions before starting the pipeline- so adjust these parameters and tell the program which data is going to be analyzed. Call this variable: "Data_conds"
makes life maybe easier. 
Cue Trading: [('clickRate (Hz)',1, 100, 900]),[ ('ITD (ms)',[-120, -80, -40, 0, 40, 80, 120]), ('ILD',[-4, -1, 0, 1, 4])]
Env Vs Pt:  [('duration (s)',[0.01, 0.05, 0.2]), ('clickRate (Hz)',[900, 4500]), ('ITD (ms)',[-0.1, 0, 0.1]), ('env ITD (ms)',[-0.1, 0, 0.1])]
Weighting: ?



'''

# specify the stim params that were permuted to generate different stimulus conditions
stimVarParams = ['duration (s)', 'clickRate (Hz)', 'ITD (ms)', 'env ITD (ms)']


dataRoot = '/ephys/2/0_ENVvsPT/Data'
analysisRoot = '/ephys/2/0_ENVvsPT/Analysis/Salt_Beaker_CSV_Files'
# analysisRoot='/ephys/2/0_ENVvsPT/Analysis/Debugging_csv_files'
figureRoot = '/ephys/2/0_ENVvsPT/Figure'

# %
'''
Read in & write in
'''

def readEphysFileFromRow(dataset, row, loadOrig=False, loadAMUA=False, loadClean=False):
    if loadOrig == True:
        filename = dataset.fileName[row]
    if loadAMUA == True:
        filename = dataset.fileName[row][:-5]+'_AMUA'  
        # filename = dataset.fileName[row][:-6]+'_AMUA'  --> ERROR occured for Salt_Beaker, had to change to -5
    if loadClean == True:
        filename = dataset.fileName[row][:-5]+'_Clean'
        # filename = dataset.fileName[row][:-6]+'_Clean'  --> ERROR occured for Salt_Beaker, had to change to -5
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
# %
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


# %
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


# %
'''
variance explained ratio calculation     
'''


def calcVarianceExplained(df):
    ss_t = ssTotalCalc(df)
    model = ols('AMUA ~ C(PT_ITD)+C(ENV_ITD)+C(PT_ITD):C(ENV_ITD)',
                data=df).fit()
    result = sm.stats.anova_lm(model, type=1)
    # ss_explainedTotal = result.iloc[0, 1]+result.iloc[1, 1]            #For EVE?!
    # ss_pt = result.iloc[0, 1]/ss_explainedTotal            #For EVE?!
    # ss_env = result.iloc[1, 1]/ss_explainedTotal            #For EVE?!
    ss_pt = result.iloc[0, 1]/ss_t
    ss_env = result.iloc[1, 1]/ss_t
    return ss_pt, ss_env


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


# %
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


# %
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
        # ax.text(x.iloc[i]*100, y.iloc[i]*100, text+'_ch'+str(channel.iloc[i]), alpha = 0.4, ha = 'center', rotation = 45)
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



# %
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
#             idxs_prev = ep.stimIndecesFor(stimPar, list(nup.keys()), stim)

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
            idxs_prev = ep.stimIndecesFor(
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


# % VE calc
'''
variance explained calculation
'''
# read in datafileInventory.csv and recordingsiteInventory.csv
# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_V3.csv')
dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_Salt_Beaker_Agar.csv')
# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_Debugging.csv')
# recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventoryV4.csv')
recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventory_Salt_Beaker_Agar.csv')
# recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventory_Debugging.csv')
# rejected_art = pd.read_csv(analysisRoot+'/LeftoverArtefact.csv')
# %%
'''
Defs needed for Salt_Beaker_Test
'''
def checkSwpsSize(swps, stimPar, stimVarParams):
    # for HK data, some set recorded by 64 channel electrode containing IC and AC, only take first 32ch(IC)
    # for FR data, some set last swp is missed or shorter than normal, replace by second last swp
    if swps[1].signal.shape[1] != 32:
        for swp in swps:
            swp.signal = swp.signal[:, :32]
    if len(swps) == len(stimPar):
        # If last swp not recorded
        Last_conds = findLastCond(stimPar, stimVarParams)
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
        # print(Short_swp)
        Short_conds = findConditionsForswp(Short_swp+1, stimPar, stimVarParams)
        # print(Short_conds)
        Last_swpIdxs = findSweepsFor(Short_conds, stimPar)
        for i, Last_swpIdx in enumerate (Last_swpIdxs):
            if Last_swpIdx == Short_swp+1:
                try:
                    swps[Last_swpIdxs[i]] = swps[Last_swpIdxs[i - 1]]
                except: 
                    swps[Last_swpIdxs[i]] = swps[Last_swpIdxs[i + 1]]
    return swps, stimPar    

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

def checkSwpsSize(swps, stimPar, stimVarParams):
    # for HK data, some set recorded by 64 channel electrode containing IC and AC, only take first 32ch(IC)
    # for FR data, some set last swp is missed or shorter than normal, replace by second last swp
    if swps[1].signal.shape[1] != 32:
        for swp in swps:
            swp.signal = swp.signal[:, :32]
    if len(swps) == len(stimPar):
        # If last swp not recorded
        Last_conds = findLastCond(stimPar, stimVarParams)
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
        # print(Short_swp)
        Short_conds = findConditionsForswp(Short_swp+1, stimPar, stimVarParams)
        # print(Short_conds)
        Last_swpIdxs = findSweepsFor(Short_conds, stimPar)
        for i, Last_swpIdx in enumerate (Last_swpIdxs):
            if Last_swpIdx == Short_swp+1:
                try:
                    swps[Last_swpIdxs[i]] = swps[Last_swpIdxs[i - 1]]
                except: 
                    swps[Last_swpIdxs[i]] = swps[Last_swpIdxs[i + 1]]
    return swps, stimPar    

def findLastCond(stimPar, stimVarParams):
    aStim=stimPar.iloc[len(stimPar)-1]
    Last_conds=[]
    for aCond in stimVarParams:
        Last_conds.append( (aCond, f'{aStim[aCond]}') )
    return Last_conds


def readEphysFileFromRow(dataset, row, stimVarParams = stimVarParams, loadOrig = False, loadAMUA = False, loadClean = False):
    if loadOrig == True:
        filename = dataset.fileName[row]
    if loadAMUA == True:
        filename = dataset.fileName[row][:-5]+'_AMUA'
    if loadClean == True:
        filename = dataset.fileName[row][:-5]+'_Clean'
    if not any('Cleaned' in col for col in dataset.columns):# if no 'Cleaned' column in csv file, read in all row
        swps, stimPar = ep.readEphysFile(dataRoot+'/'+dataset.animDir[row]+'/'+filename)
        swps, stimPar = checkSwpsSize(swps, stimPar, stimVarParams)
        print(filename)
    if any('Cleaned' in col for col in dataset.columns):# if exist 'Cleaned' column in csv file, only read in row labed as '1' in cleaned
        if dataset.Cleaned[row] == 1:
            swps, stimPar=ep.readEphysFile(dataRoot+'/'+dataset.animDir[row]+'/'+filename)
            swps, stimPar = checkSwpsSize(swps, stimPar, stimVarParams)
            print(filename)
        if dataset.Cleaned[row] == 0:
            swps = []
            stimPar = []
    return swps, stimPar

def writeEphysFileFromRow(dataset, row, swps, stimPar, writeAMUA = False, writeClean = False):
    if writeAMUA == True:
        filename = dataset.fileName[row][:-5]+'_AMUA'
    if writeClean == True:
        filename = dataset.fileName[row][:-5]+'_Clean'
    ep.writeEphysFile(dataRoot+'/'+dataset.animDir[row]+'/'+filename, swps, stim = stimPar)

def writeCsvFileFromRow(dataset, row, writeWilcoxon = False):
    if writeWilcoxon == True:
        filename = dataset.fileName[row][:-5]+'.csv'
    dataset.to_csv(dataRoot+'/'+dataset.animDir[row]+'/'+filename, index=False)
    

#%%

for row in range(len(dataFiles)):
    # if dataFiles.loc[row, 'condition'] in ['ITD+']:
    # if row == 0:
    # if row in range(59, 60):
    # if True:
        if dataFiles.Cleaned[row] == 0:
            continue
        fname = dataFiles.fileName[row]
        csvName = fname[:-5]+'_Clean.csv'
        animDir = dataFiles.animDir[row]
        Swps_norm, stimPar_norm = readEphysFileFromRow(dataFiles, row, loadOrig = True)
        swpAMUA, stimParAMUA = readEphysFileFromRow(dataFiles, row, loadAMUA = True)
        stimParCLEAN = pd.read_csv(dataRoot+'/'+animDir+'/'+csvName)
        # swpCLEAN, stimParCLEAN = readEphysFileFromRow(dataFiles, row, loadClean = True)

        # from datafilename find corresponding recording sites in recordingsiteInventory.csv
        sub_recordingSites = findRecordingSite(dataFiles, row)

        #Check the sweeps with leftover artefact:
        filename = dataFiles.iloc[row]['fileName']

        nup = ep.nonUniqueParams(stimPar_norm)
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

                        idxs_prev = ep.stimIndecesFor(stimParCLEAN, list(nup.keys()), stimulus)
                        #from 0 to 1079


                        #check if there is residual artefact and remove single trials if less more than 1/4 still good:
                        idxs = remove_single_ARTsweeps (idxs_prev, channel, stimParCLEAN)
                        # CHECKING.append(idxs+1)

                        signal = poolChannel(swpAMUA[idxs+1], channel)[:int(stimulus[0]*swpAMUA[0].sampleRate), :].mean(axis = 0) #Warning: idxs = idxs -1 originally due to ep.stimIndecesFor

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
recordingSites.to_csv(analysisRoot+'/'+'recordingsiteInventory_Salt_Beaker_Agar.csv', index=False)


#%%
# # Check AMUA responses:
# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_Salt_Beaker.csv')
# recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventory_Salt_Beaker_VE.csv')

# #     # if dataFiles.loc[row, 'condition'] in ['ITD+']:
# # row = 59
# #     # if row in range(59, 60):
# #     # if True:
# # if dataFiles.Cleaned[row] == 0:
# #     print('No Bueno')
# # fname = dataFiles.fileName[row]
# # csvName = fname[:-6]+'_Clean.csv'
# # animDir = dataFiles.animDir[row]
# # swpAMUA, stimParAMUA = readEphysFileFromRow(dataFiles, row, loadAMUA = True)
# # stimParCLEAN = pd.read_csv(dataRoot+'/'+animDir+'/'+csvName)


# # dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_V3.csv')

# corhort = 'SB'
# animNum = 0
# penNum = 0
# chanNum = 15
# duration = 0.2
# clickRate = 4500

# anim = dataFiles[(dataFiles['condition'] == corhort)]['animDir'].unique()[animNum]
# filename = dataFiles[(dataFiles['animDir'] == anim) & (dataFiles['Cleaned'] == 1)].iloc[penNum]['fileName']
# print(filename)
# # swpsClean, stimParCLEAN = readEphysFile(dataRoot+'/'+anim+'/'+filename[:-6]+'_Clean')
# swpsClean, stimParCLEAN = readEphysFileFromRow(dataFiles, animNum, loadClean = True)
# # swpsAMUA, stimPar = readEphysFile(dataRoot+'/'+anim+'/'+filename[:-6]+'_AMUA')
# swpsAMUA, stimParAMUA = readEphysFileFromRow(dataFiles, animNum, loadAMUA = True)

# # plot3X3plot_Sanity(swpsAMUA, sub_conditions = [duration, clickRate])


# ITD=[-0.1, 0, 0.1]

# for itd_pt in ITD: 
#     for itd_env in ITD:
#         # print(itd_pt, itd_env)
#         conditions = [('duration (s)', duration), ('clickRate (Hz)', clickRate), ('ITD (ms)', itd_pt), ('env ITD (ms)', itd_env)]
        
#         plot32Channel_Sanity(swpsAMUA, conditions, stimParCLEAN, allRepeats=True, avgRepeats=False)
# # signal = swpsAMUA[1], chanNum
# # plt.plot(signal)

# # from datafilename find corresponding recording sites in recordingsiteInventory.csv
#%%

'''
plot the left over sweeps that are used for VE calculation since we check for leftover artefacts:
'''

def plot3X3plot_Sanity(swps, sub_conditions, filename, rejected_art, chanNum, itd=[-0.1, 0, 0.1], allRepeats=False, avgRepeats=False):
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

def plot32Channel_Sanity(swps, conditions, stimParCLEAN, allRepeats=False, avgRepeats=False, ncols=8, nrows=4):
    sampleRate = swps[1].sampleRate
    Duration = conditions[0][1]
    t = np.arange(0, Duration+0.2, 1/sampleRate)
    nchans = swps[1].signal.shape[1]
    idxs_prev = findSweepsFor(conditions.copy(), stimPar)
    # print(idxs_prev)
    # swpPlot = swps[idxs_prev]
    
    fig, axs = prepare4X8Plot()
    for cc in range(nchans):
        ax = axs[cc]
        
        swpIdx = remove_single_ARTsweeps (idxs_prev, cc, stimParCLEAN)
        print(swpIdx)
       
        if len(swpIdx) < 15:
            print('Chan bad:', conditions, cc)
            continue
        
        swpPlot = swps[swpIdx]
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
# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_V3.csv')
# recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventoryV2_not_clean_check.csv')
# recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventoryV3.csv')
# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_Salt_Beaker_Agar.csv')
# recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventoryV2.csv')
dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_Debugging.csv')
recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventory_Debugging.csv')
# recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventory_Salt_Beaker_Agar.csv')




corhorts = dataFiles.condition.unique()
for corhNum, corhort in enumerate(corhorts):
    if corhNum == 0:
        fig, axs = prepare2X3Plot()
        anims = dataFiles[dataFiles['condition']
                          == corhort]['animDir'].unique()
        print(anims)
        unitNum = 0
        for animNum, anim in enumerate(anims):
            if anim == "2023_02_23" or anim == "2023_02_23":
                continue

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
                     va='center', rotation=-90, fontsize=16)
            fig.text(0.38, 0.053, 'Variance explained by ENV_ITD (%)',
                     va='center', fontsize=16)
            fig.suptitle(corhort+'_ResponsiveUnit#'+str(unitNum), fontsize=18)
#             # plt.savefig('/ephys/2/0_ENVvsPT/Figure/Result_ScatterPlot/'+corhort
            
# %%
'''
VE distribution plot, penetration label by different color
'''
# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_not_clean_check.csv')
# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_V3.csv')
# recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventoryV2_not_clean_check.csv')
# recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventoryV3.csv')
# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory.csv')
# recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventoryV2.csv')

corhorts = dataFiles.condition.unique()
colors = plt.cm.tab20(np.random.rand(40))
for corhNum, corhort in enumerate(corhorts):
    # if corhNum == 0:
    if True:
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
# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_V3.csv')
# recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventoryV2_not_clean_check.csv')
# recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventoryV3.csv')
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
            if animNum == 6:
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
# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_V3.csv')
# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_Salt_Beaker_Agar.csv')
# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_Salt_Beaker_Agar.csv')
dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_Debugging.csv')


corhort = 'NH'
animNum = 0
penNum = 0
chanNum = 0
duration = 0.2
clickRate = 4500

anim = dataFiles[(dataFiles['condition'] == corhort)]['animDir'].unique()[animNum]
filename = dataFiles[(dataFiles['animDir'] == anim) & (dataFiles['Cleaned'] == 1)].iloc[penNum]['fileName']
print(filename)
swpsClean, stimParCLEAN = ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-5]+'_Clean')
swpsAMUA, stimPar = ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-5]+'_AMUA')
# swps, stimPar = ep.readEphysFile(dataRoot+'/'+anim+'/'+filename)
# %% plot clean data

# Cut_Window_sec = 0.002
# sampleRate = swpsClean[0].sampleRate
# StepSize_sec = 1/sampleRate
# Samples_2_cut = int(Cut_Window_sec/StepSize_sec)


# #Use the Rz2ephys preprogrammed methods for that! 
# #However, this method interpolates the values from 0 to the value you want to cut- in our case here it is Samples_2_cut
# #swpsILD[0].interpBlanking(0,Samples_2_cut)
# for i in range (len (swpsClean)):
#     swpsClean[i].interpBlanking(0,Samples_2_cut)


for chanNum in range(2):
# if True:
    sampleRate = swpsClean[1].sampleRate
    # sampleRate = swpsAMUA[1].sampleRate
    t = np.arange(0, duration+0.2, 1/sampleRate)
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
        idxs_prev = ep.stimIndecesFor(stimParCLEAN, keys, conds)

        # idxs_prev = idxs_prev + 1
        # signal = poolChannel(swpsClean[idxs_prev], chanNum)

        swpIdx = remove_single_ARTsweeps(idxs_prev, chanNum, stimParCLEAN)
        swpIdx = swpIdx + 1
        
        # signal = poolChannel(swpsClean[idxs_prev+1], chanNum)
        signal = poolChannel(swpsClean[swpIdx], chanNum)
        # signal = poolChannel(swps[swpIdx], chanNum)
        # signal = poolChannel(swpsAMUA[swpIdx], chanNum)
                
        signal = (signal-signal[0, :])[:len(t)]
        # signal = (signal-signal[0, :])[:len(t)].mean(axis = 1)
        # signal = signal[:len(t)].mean(axis = 1)
        # signal = signal-signal[0]
        ax.plot(t, signal)
        ax.axvline(x=duration, color='k', linestyle='--')
        ax.set_title(chanNum)
        # ax.set_ylim(-3e-6, 7e-6)
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
import re
import matplotlib.patches as patches
figurepath = '/home/colliculus/projects/Manuscripts/ENVvsPT_Ephys/'
figurepath = '/home/colliculus/projects/Manuscripts/ENVvsPT_Ephys/'

analysisRoot = '/ephys/2/0_ENVvsPT/Analysis'
dataRoot = '/ephys/2/0_ENVvsPT/Data'
dataFiles = pd.read_csv(analysisRoot+'/datafileInventory.csv')
# dataFiles = pd.read_csv(analysisRoot+'/Salt_Beaker_CSV_Files/datafileInventory_Salt_Beaker_Agar.csv')
# baselinerecordingSites = pd.read_csv(analysisRoot+'/Salt_Beaker_CSV_Files/recordingsiteInventory_Salt_Beaker_Agar_baseline.csv')
# recordingSites = pd.read_csv(analysisRoot+'/Salt_Beaker_CSV_Files/recordingsiteInventory_Salt_Beaker_Agar.csv')
baselinerecordingSites = pd.read_csv('/ephys/2/0_ENVvsPT/Analysis/recordingsiteInventory_baseline.csv')
recordingSites = pd.read_csv('/ephys/2/0_ENVvsPT/Analysis/recordingsiteInventory.csv')

# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_V3.csv')

# str_recordingsites = analysisRoot+'/recordingsiteInventory_EVE.csv'
str_recordingsites = analysisRoot+'/recordingsiteInventory.csv'


# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_V3.csv')
# dataFiles = pd.read_csv(analysisRoot+'/drecordingsiteInventory_Salt_Beaker_VE.csv')


# str_recordingsites = analysisRoot+'/recordingsiteInventory_EVE.csv'
# str_recordingsites = analysisRoot+'/recordingsiteInventoryV3.csv'
# str_recordingsites = analysisRoot+'/datafileInventory_Salt_Beaker_V2.csv'

recordingSites = pd.read_csv(str_recordingsites)


# Function to check if 'EVE' is in the string using regular expressions
def contains_eve(input_string):
    # Regular expression to search for 'EVE' (case-sensitive)
    pattern = r'_EVE'
    
    # Use re.search to check if 'EVE' is found in the input string
    if re.search(pattern, input_string):
        return True
    else:
        return False

corhorts = dataFiles.condition.unique()
corhorts = corhorts[:3]

animDir = dataFiles.animDir.unique()
# Get the headers
headers = recordingSites.columns
Data_header = headers[4:]       ################Careful here- old code was 6: !!!
significance_headers = [col for col in Data_header if "significance" in col]
VE_headers = [
    col for col in Data_header if "significance" not in col and "pValue" not in col]
pValue_headers = [col for col in Data_header if "pValue" in col]

# jusit a quick and dirty way to get the headers for a plot...
Headers_Plot = []
# This is to get the VE letters --> Jan wated that for the plot
VE_kind = [header.split('_', 2)[-1] for header in VE_headers]
VE_kind = [f"VE {entry[2:].upper()}" for entry in VE_kind]
Dur_kind = [header.split('_', 1)[0] for header in VE_headers]
pps_kind = [header.split('_')[1] for header in VE_headers]


for i in (range(len(VE_headers))):
    # print (head)

    Headers_Plot.append(VE_kind[i] + ' ' +
                        Dur_kind[i][3:] + ' s, ' + pps_kind[i][4:-2] + ' pps')

Anims = []
for anim in animDir:
    # print(anim)
    filenames = dataFiles[dataFiles['animDir'] == anim]['fileName']
    sub_recordingSites = []
    for filename in filenames:
        sub_recordingSites.append(
            recordingSites[recordingSites['fileName'] == filename])
    sub_recordingSites = pd.concat(sub_recordingSites, ignore_index=True)
    # sub_recordingSites = sub_recordingSites[sub_recordingSites['Responsive'] == 1]
    Anims.append(sub_recordingSites)

Anims_baseline = []
for anim in animDir:
    filenames = dataFiles[dataFiles['animDir'] == anim]['fileName']
    sub_recordingSites_baseline = []
    for filename in filenames:
        sub_recordingSites_baseline.append(
            baselinerecordingSites[baselinerecordingSites['fileName'] == filename])
    sub_recordingSites_baseline = pd.concat(sub_recordingSites_baseline, ignore_index=True)
    # sub_recordingSites = sub_recordingSites[sub_recordingSites['Responsive'] == 1]
    Anims_baseline.append(sub_recordingSites_baseline)

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
    
    # Vertical
    # fig_box, axes_box = plt.subplots (3, 1, figsize = (8, 12), sharex = True)
    # # plt.tight_layout(rect = [0, 0.1, 1, 0.9])
    # fig_box.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.13)
    # # make every second background pale 
    # highlighted_groups = [(3,4), (7,8), (11,12)]
    # for i, ax in enumerate (axes_box):
    #     ax.set_ylim(-5, 100)
    #     for start, end in highlighted_groups:
    #         ax.add_patch(
    #             patches.Rectangle(
    #                 (start - 0.5, -5),
    #                 end - start + 1, 
    #                 105, 
    #                 facecolor = "palegoldenrod",
    #                 alpha = 0.25,
    #                 zorder = 0
    #             )
    #         )

    # Horizontal
    fig_box, axes_box = plt.subplots (1, 3, figsize = (8.1, 6), sharex = True)
    # plt.tight_layout(rect = [0, 0.1, 1, 0.9])
    fig_box.subplots_adjust(right=0.98, wspace = 0.12)
    # make every second background pale 
    highlighted_groups = [(2.5,3.5), (6.5,7.5), (10.5,11.5)]
    flierprops = dict(marker = 'o', markersize = 0.2, color = 'black', alpha = 0.7)
    for i, ax in enumerate (axes_box):
        ax.set_xlim(-5, 100)
        for start, end in highlighted_groups:
            ax.add_patch(
                patches.Rectangle(
                    (-5, start, -0.5),
                    105, 
                    end - start + 1, 
                    facecolor = "palegoldenrod",
                    alpha = 0.25,
                    zorder = 0
                )
            )    
            
    # fig_box.subplots_adjust(hspace=0.2)
    
    #Exclude the other subplots 
    fig_bar, axes_bar = plt.subplots (3, 1, figsize = (8, 12), sharex = True)

    fig_6_bar, axes_6_bar = plt.subplots (2, 3, figsize = (7.2, 6), sharey = True)
    
    if contains_eve(str_recordingsites):
        fig_hist, axes_hist = plt.subplots (3, 6, figsize = (18, 10), sharex = True)

count_outside = 0

for corhNum, corhort in enumerate(corhorts):
    # if type(Onlycohort) == int :
    # if corhNum != Onlycohort and OnlycohortStr != 'n' and OnlycohortStr != 'all':
    if isinstance (Onlycohort, int ) and corhNum != Onlycohort:
        continue
        # plt.figure()
    # print(corhort)
    # all_units = []
    All_data = [] #holds all animals penetrations with all VE headers
    Sig_data = [] #holds the significance vals 0 or 1
    Baseline_Sig_data = []
    No_Animals = 0
    # All_labels = []
    for idx, anim in enumerate(Anims):
        # print(idx)
        # if idx == 22 or idx == 19 or idx == 13:
            # continue
        # print (Anims[idx].condition.iloc[0])
        if Anims[idx].cohort.iloc[0] == corhort:
            No_Animals += 1
            # print(corhort)
            # Check here how ,many UNITS!

            # Units = len(anim)
            # all_units.append(Units)
            All_data.append(anim[VE_headers]*100)
            Sig_data.append(anim[significance_headers])
    
    for idx_base, anim_base in enumerate(Anims_baseline):
        if Anims_baseline[idx_base].cohort.iloc[0] == corhort:
            Baseline_Sig_data.append(anim_base[significance_headers])


            # All_labels.append(Headers_Plot)
            
            
    ######################################################
    ######## Remove if baseline VE is significant ########
    ######################################################
    # --> Changes the values with ones to a zero 
    
    Baseline_list = []
    for i, pandas in enumerate (Baseline_Sig_data):
        nested_indices = []
        for i in range(0, len(pandas.columns), 2):
            col_pair = pandas.columns[i:i+2]
            mask = (pandas[col_pair[0]] == 1) | (pandas[col_pair[1]] == 1)
            indices = pandas.index[mask].tolist()
            nested_indices.append(indices)
        Baseline_list.append(nested_indices)
    
    for i, pandas in enumerate (Baseline_Sig_data):
        print(i)
        for column, index_remove in enumerate (Baseline_list[i]):
            # print(column) #--> 0-5
            # print(index_remove) #--> liste removed indices
            LoopJump = column * 2
            cols = Baseline_Sig_data[i].columns[LoopJump:LoopJump+2]
            # print (cols) #--> header list - each header and the following.
            # Now replace these values with nan and do the same for all data --> solved :) 
            Sig_data[i].loc[index_remove, cols] = np.nan
            # All_data[i].loc[index_remove, cols] = np.nan
        
    for i, pandas in enumerate (Baseline_Sig_data):
        # print(i)
        for column, index_remove in enumerate (Baseline_list[i]):
            # print(column) #--> 0-5
            # print(index_remove) #--> liste removed indices
            LoopJump = column * 2
            cols = All_data[i].columns[LoopJump:LoopJump+2]
            # print (cols) #--> header list - each header and the following.
            # Now replace these values with nan and do the same for all data --> solved :) 
            All_data[i].loc[index_remove, cols] = np.nan

    #New counting function also:
    all_units = []
    for i, pandas in enumerate (Sig_data):
        rows_with_entries = pandas.notna().any(axis=1).sum()
        all_units.append(rows_with_entries)
    No_All_Units = np.sum(all_units)    

    ############################################################################################
    ############################################################################################
    ############################################################################################
    
    
    
    
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
    # No_All_Units = np.sum(all_units)

    #################################### Plot for comparison of single cohort data and single anims data ####################################
    if not OnlycohortStr == 'all':
        
        #################################### Make Boxplots for either one animal or one cohort ####################################
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
                    #plot percentages here if needed
                    # plt.text(
                    #     i_Sig + 1, -1, f"Units : {n}", ha='center', fontsize=10, color='blue')
                    # plt.text (i_Sig + 1, -2, f"Units2 : {n2}", ha='center', fontsize=10, color='blue')
                    # plt.text(
                    #     i_Sig + 1, -3, f"% : {Percentage}", ha='center', fontsize=10, color='blue')
    
    
        plt.xticks(rotation=45)
        # plt.tight_layout()
        plt.axvline(x=2.5, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=4.5, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=6.5, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=8.5, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=10.5, color='black', linestyle='--', linewidth=1)
        
        if corhort == 'ITD+':
            corhort = 'ND + CI chronic'
        if corhort == 'ND':
            corhort = 'ND + acute CI'
        if corhort == 'NH':
            corhort = 'AD + acute CI'
        
        title = f"{corhort}, All units: {No_All_Units} , No. animals: {No_All_Units}"
        plt.title(title)
        
        if Singleanim:
            title = f"{corhort}, All units: {No_All_Units} , No. animal: {anim_check}"
            plt.title(title)
            
        if contains_eve(str_recordingsites):
            plt.ylabel('EVE Values')
        else:
            plt.ylabel('VE Values')
        plt.show()
    
    
        ##################################### Barplot from here for single animals or cohorts ####################################
        plt.figure()
        cols = ['blue', 'purple'] * (len(Headers_Plot) // 2)  # Create a list with alternating green, purple
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
        # plt.tight_layout()
        plt.axvline(x=1.5, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=3.5, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=5.5, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=7.5, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=9.5, color='black', linestyle='--', linewidth=1)
        plt.show()


    #################################### Plot for comparison of all data ####################################
    List_of_Cols = [[0.2, 0.1, 0.4], [0.2, 0.5, 0.6], [0.1, 0.0, 1]]
    AD_Acute_Color_Box = [[0.2, 0.1, 0.4], [0.2, 0.1, 0.4], [0.2, 0.1, 0.4], [0.2, 0.1, 0.4], [0.2, 0.1, 0.4], [0.2, 0.1, 0.4], [0.2, 0.1, 0.4], [0.2, 0.1, 0.4], [0.2, 0.1, 0.4], [0.2, 0.1, 0.4], [0.2, 0.1, 0.4], [0.2, 0.1, 0.4]]
    ND_Acute_Color_Box = [[0.2, 0.5, 0.6], [0.2, 0.5, 0.6], [0.2, 0.5, 0.6], [0.2, 0.5, 0.6], [0.2, 0.5, 0.6], [0.2, 0.5, 0.6], [0.2, 0.5, 0.6], [0.2, 0.5, 0.6], [0.2, 0.5, 0.6], [0.2, 0.5, 0.6], [0.2, 0.5, 0.6], [0.2, 0.5, 0.6]]
    ND_Chronic_Color_Box = [[0.1, 0.0, 1],  [0.1, 0.0, 1],  [0.1, 0.0, 1],  [0.1, 0.0, 1],  [0.1, 0.0, 1],  [0.1, 0.0, 1],  [0.1, 0.0, 1],  [0.1, 0.0, 1],  [0.1, 0.0, 1],  [0.1, 0.0, 1],  [0.1, 0.0, 1],  [0.1, 0.0, 1]]
    List_of_Cols_Box = [AD_Acute_Color_Box, ND_Acute_Color_Box, ND_Chronic_Color_Box]
    
    AD_Acute_Color = [0.2, 0.1, 0.4]
    ND_Acute_Color = (0.2, 0.5, 0.6)
    ND_Chronic_Color = (0.1, 0.0, 1)
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
                    #Plot percentage units here if neede:
                    # plt.text(
                    #     i_Sig + 1, -1, f"Units : {n}", ha='center', fontsize=10, color='blue')
                    # # plt.text (i_Sig + 1, -2, f"Units2 : {n2}", ha='center', fontsize=10, color='blue')
                    # plt.text(
                    #     i_Sig + 1, -3, f"% : {Percentage}", ha='center', fontsize=10, color='blue')
        
        #################################### Plo violin data for EVE vals ####################################
        violin_data = Combined_Data[::2]
        
        if contains_eve(str_recordingsites):
            for i_box, ax_box in enumerate (axes_box):
                if i_box == corhNum:
                    parts = ax_box.violinplot(violin_data, showmeans = False, showextrema = False)
                    
                    for pc in parts['bodies']:
                        pc.set_facecolor(List_of_Cols[i_box])  # Set the face color to brown (RGB or hex)
                        pc.set_edgecolor('black')  # Set the edge color to black
                        pc.set_alpha(0.7)  # Set transparency, 0.0 is fully transparent, 1.0 is fully opaque
                    
                    if contains_eve(str_recordingsites):
                        ax_box.axvline(x=1.5, color='black', linestyle='--', linewidth=1)
                        ax_box.axvline(x=2.5, color='black', linestyle='--', linewidth=1)
                        ax_box.axvline(x=3.5, color='black', linestyle='--', linewidth=1)
                        ax_box.axvline(x=4.5, color='black', linestyle='--', linewidth=1)
                        ax_box.axvline(x=5.5, color='black', linestyle='--', linewidth=1)                

                    ax_box.tick_params(axis = 'y', labelsize = 13)
                    

                    if corhort == 'ITD+':
                        corhort = 'ND + chronic CI'
                    if corhort == 'ND':
                        corhort = 'ND + acute CI'
                    if corhort == 'NH':
                        corhort = 'AD + acute CI'
                    
                    title = f"{corhort}, All units: {No_All_Units} , No. animals: {No_Animals}"
                    ax_box.set_title(title)
                
                    if contains_eve(str_recordingsites):
                        ax_box.set_ylabel('Significant EVE Values', fontsize = 14)
                    
                    if i_box == 2:
                        ax_box.set_xticks([1, 2, 3, 4, 5, 6])
                        EVE_Headers_Plot = []
                        for i in Headers_Plot:
                            EVE_Headers_Plot.append("E" + i)
                        ax_box.set_xticklabels(EVE_Headers_Plot[::2], rotation=45)
                

                    plt.subplots_adjust(bottom=0.3)     
                    # plt.tight_layout()
                    plt.show()
                
        ##################################### Plot the histogram instead of violin for EVE vals ####################################
        if contains_eve(str_recordingsites):
            # axes_hist = axes_hist.flatten()
            for i_hist, ax_hist in enumerate (axes_hist):
                # print(i_hist)
                binsize = 10
                # count_outside += len(violin_data)
                if i_hist == corhNum:
                    for comb in range (len(violin_data)):
                        # print(comb)
                        data_hist = violin_data[comb].reset_index(drop=True)
                        axes_hist[i_hist, comb].hist(data_hist, bins = binsize, color = List_of_Cols[i_hist], density = True, edgecolor='black', alpha=0.7)              
                        # axes_hist[i_hist, comb].set_ylim(0,0.08)
                        # axes_hist[i_hist, comb].set_xlim(0,100)
                        axes_hist[i_hist, comb].tick_params(axis = 'y', labelsize = 13)
                        axes_hist[i_hist, comb].tick_params(axis = 'y', labelsize = 13)
                    
                    # if corhort == 'ITD+':
                    #     corhort = 'ND + chronic CI'
                    # if corhort == 'ND':
                    #     corhort = 'ND + acute CI'
                    # if corhort == 'NH':
                    #     corhort = 'AD + acute CI'
                    # title = f"{corhort}, All units: {No_All_Units} , No. animals: {No_Animals}"
                    # axes_hist[i_hist, comb].set_title(title)
                    
                    
                    
            for i, ax in  enumerate(axes_hist.flat):
                # print(i)
                
                if i < 6:
                    ax.set_title(EVE_Headers_Plot[::2][i], fontsize = 15)
                
                ax.set_xlabel ('Percentage of significance for EVEpt', fontsize = 13)
                ax.set_ylabel ('Percentage of Units', fontsize = 15)
                
                if i != 0 and i != 6 and i != 12:
                    ax.set_ylabel('')  # Removes the x-axis label
                
                if i < 12:
                    ax.set_xlabel('')  # Removes the y-axis label
            # for ax in  axes_hist.flat:
            #     ax.label_outer()
            

            
                # plt.tight_layout()
                custom_leg = [
                    Line2D([0], [0], color= AD_Acute_Color, lw=8, label='AD + acute CI'),  
                    Line2D([0], [0], color= ND_Acute_Color, lw=8, label='ND + acute CI'), 
                    Line2D([0], [0], color= ND_Chronic_Color, lw=8, label='ND + chronic CI') 
                    ]
                
                
                axes_hist[0, 0].legend(handles=custom_leg, loc='upper left', fontsize=12)
                plt.subplots_adjust(left = None, bottom = None, right = None, top = None)
                plt.show()
                
        #################################### Plot boxplots for all animals vertically ####################################  
       
        # else:
        #     labels = [
        #         'PT, 10 ms, 900 pps', 'ENV, 10 ms, 900 pps',
        #         'PT, 10 ms, 4500 pps', 'ENV, 10 ms, 4500 pps',
        #         'PT, 50 ms, 900 pps', 'ENV, 50 ms, 900 pps',
        #         'PT, 50 ms, 4500 pps', 'ENV, 50 ms, 4500 pps',
        #         'PT, 200 ms, 900 pps', 'ENV, 200 ms, 900 pps',
        #         'PT, 200 ms, 4500 pps', 'ENV, 200 ms, 4500 pps'
        #     ]
            
        #     for i_box, ax_box in enumerate (axes_box):
        #         if i_box == corhNum:
        #             bplot = ax_box.boxplot(Combined_Data, patch_artist=True)#, labels = ['PT 10 ms, 900 pps', 'ENV 10 ms, 900 pps', 'PT 10 ms, 4500 pps', 'ENV 10 ms, 4500 pps', 
        #                                                  #   'PT 50 ms, 900 pps', 'ENV 50 ms, 900 pps', 'PT 50 ms, 4500 pps', 'ENV 50 ms, 4500 pps',
        #                                                   #  'PT 200 ms, 900 pps', 'ENV 200 ms, 900 pps', 'PT 200 ms, 4500 pps', 'ENV 200 ms, 4500 pps'], patch_artist=True)
        #             ax_box.tick_params(axis = 'y', labelsize = 13)
        #             # ax_box.tick_params(axis = 'x', labelsize = 10, rotation = 45)

        #             ax_box.axvline(x=2.5, color='black', linestyle='--', linewidth=1)
        #             ax_box.axvline(x=4.5, color='black', linestyle='--', linewidth=1)
        #             ax_box.axvline(x=6.5, color='black', linestyle='--', linewidth=1)
        #             ax_box.axvline(x=8.5, color='black', linestyle='--', linewidth=1)
        #             ax_box.axvline(x=10.5, color='black', linestyle='--', linewidth=1)
                    
        #             if corhort == 'ITD+':
        #                 corhort = 'ND + chronic CI'
        #             if corhort == 'ND':
        #                 corhort = 'ND + acute CI'
        #             if corhort == 'NH':
        #                 corhort = 'AD + acute CI'
                    
        #             title = f"{corhort}, {No_All_Units} units, {No_Animals} animals"
        #             ax_box.set_title(title)
        #             ax_box.set_ylim(-5, 100)

        #             ax_box.set_ylabel('VE values (in %)', fontsize = 14)
                        
        #         for patch, flier, color in zip(bplot['boxes'], bplot['fliers'], List_of_Cols_Box[i_box]):
        #             # fill boxes
        #             patch.set_facecolor(color)
        #             # set flier (outlier) color to match box
        #             # flier.set(marker='o', markerfacecolor=color, markeredgecolor=color, alpha=0.6)

        #         if i_box == 2:
        #             ax_box.set_xticks(range(1, 13))  # positions (112)
        #             ax_box.set_xticklabels(labels, rotation=45,
        #                                ha='right', va='top', fontsize=10)
        #         else:
        #             ax_box.set_xticks(range(1, 13))
        #             ax_box.set_xticklabels([])  # hide labels for upper subplots


        #             if i_box == 2: 
        #                 for index, tick in enumerate(ax_box.get_xticklabels()[0:12]):
        #                     print(index)
        #                     if index < 12:
        #                         tick.set_rotation(45)
        #                         tick.set_ha('right')   # horizontal alignment
        #                         tick.set_va('top')     # optional, vertical alignment
        #                         tick.set_fontsize(10)
                                
        #         # ax_box.tick_params(axis = 'x', rotation=45, ha = 'right', labelsize = 10)
        #         # ax_box.set_xticklabels(ax_box.get_xticks(), rotation=45, ha='right', fontsize=10)
                
        #         # plt.subplots_adjust(bottom = 0.2)
        #         plt.show()
        #         # plt.savefig(figurepath + 'Comparison_Boxplot_NewFig.5.png', dpi = 300)

 #################################### Plot boxplots for all animals horizontally ####################################  

        else:
            labels = [
                'PT, 10 ms,\n 900 pps', 'ENV, 10 ms,\n 900 pps',
                'PT, 10 ms,\n 4500 pps', 'ENV, 10 ms,\n 4500 pps',
                'PT, 50 ms,\n 900 pps', 'ENV, 50 ms,\n 900 pps',
                'PT, 50 ms,\n 4500 pps', 'ENV, 50 ms,\n 4500 pps',
                'PT, 200 ms,\n 900 pps', 'ENV, 200 ms,\n 900 pps',
                'PT, 200 ms,\n 4500 pps', 'ENV, 200 ms,\n 4500 pps'
            ]
            
            for i_box, ax_box in enumerate (axes_box):
                if i_box == corhNum:
                    bplot = ax_box.boxplot(Combined_Data, patch_artist=True, vert = False, zorder = 1, flierprops = flierprops)#, labels = ['PT 10 ms, 900 pps', 'ENV 10 ms, 900 pps', 'PT 10 ms, 4500 pps', 'ENV 10 ms, 4500 pps', 
                                                         #   'PT 50 ms, 900 pps', 'ENV 50 ms, 900 pps', 'PT 50 ms, 4500 pps', 'ENV 50 ms, 4500 pps',
                                                          #  'PT 200 ms, 900 pps', 'ENV 200 ms, 900 pps', 'PT 200 ms, 4500 pps', 'ENV 200 ms, 4500 pps'], patch_artist=True)
                    ax_box.tick_params(axis = 'y', labelsize = 13)
                    # ax_box.tick_params(axis = 'x', labelsize = 10, rotation = 45)
        
                    
                    # Y lines
                    ax_box.axhline(y=2.5, color='black', linestyle='--', linewidth=1)
                    ax_box.axhline(y=4.5, color='black', linestyle='--', linewidth=1)
                    ax_box.axhline(y=6.5, color='black', linestyle='--', linewidth=1)
                    ax_box.axhline(y=8.5, color='black', linestyle='--', linewidth=1)
                    ax_box.axhline(y=10.5, color='black', linestyle='--', linewidth=1)
                    
                    if corhort == 'ITD+':
                        corhort = 'ND + chronic CI'
                    if corhort == 'ND':
                        corhort = 'ND + acute CI'
                    if corhort == 'NH':
                        corhort = 'AD + acute CI'
                    
                    title = f"{corhort}\n{No_All_Units} units, {No_Animals} animals"
                    ax_box.set_title(title)
                    # ax_box.set_ylim(-5, 100)
        
                    ax_box.set_xlabel('VE values (in %)', fontsize = 12)
                        
                for patch, flier, color in zip(bplot['boxes'], bplot['fliers'], List_of_Cols_Box[i_box]):
                    # fill boxes
                    patch.set_facecolor(color)
                    # set flier (outlier) color to match box
                    # flier.set(marker='o', markerfacecolor=color, markeredgecolor=color, alpha=0.6)
        
                if i_box == 0:
                    ax_box.set_yticks(range(1, 13))  # positions (112)
                    ax_box.set_yticklabels(labels, fontsize = 9.8)
                else:
                    ax_box.set_yticks(range(1, 13))
                    ax_box.set_yticklabels([])  # hide labels for upper subplots
        

                # plt.subplots_adjust(bottom = 0.2)
                plt.show()
                plt.savefig(figurepath + 'Comparison_Boxplot_NewFig.5_Horizontally.png', dpi = 300)

        ##################################### Barplot with each cohort seperately in a subplot (3,1) ####################################
        # plt.figure()
        
        cols = ['blue', 'purple'] * (len(Headers_Plot) // 2)  # Create a list with alternating green, purple
        for i_bar, ax_bar in enumerate (axes_bar):
            if i_bar == corhNum:
                for ind_head, Head in enumerate (Headers_Plot):
                    bars = ax_bar.bar(Head, Percentage_vals[ind_head], color = cols[ind_head])
                    title = f"{corhort}, All units: {No_All_Units} , No. animals: {No_Animals}"
                    ax_bar.set_title(title, fontsize = 17)
                    ax_bar.set_ylim(0,100)
                    ax_bar.tick_params(axis = 'y', labelsize = 13)
                    ax_bar.tick_params(axis = 'x', labelsize = 13)
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
                Line2D([0], [0], color='blue', lw=8, label='Pt'),  
                Line2D([0], [0], color='purple', lw=8, label='Env') 
                ]
        
            
            plt.subplots_adjust(bottom = 0.2)
            axes_bar[2].legend(handles=custom_legend, loc='upper right', fontsize=12)
            
            if contains_eve(str_recordingsites):
                EVE_Headers_Plot = []
                for i in Headers_Plot:
                    EVE_Headers_Plot.append("E" + i)
                ax_bar.set_xticklabels(EVE_Headers_Plot, rotation=45)
            
            else:
                ax_bar.tick_params(axis = 'x', rotation=45, labelsize = 10)
            
            plt.subplots_adjust(bottom = 0.25)
            plt.tight_layout()
            plt.show()
    
    
    
        ##################################### Barplot with each cohort seperately in a 6er subplot (2,3) ####################################
        # plt.figure()
        for i_6_bar, ax_6_bar in enumerate (axes_6_bar.flatten()):
            # print(i_6_bar)
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
            
            modified_headers_6_bar = [headers_6_bar[0], headers_6_bar[2], headers_6_bar[4], headers_6_bar[1], headers_6_bar[3], headers_6_bar[5]]
            modified_percentage_6_bar = [percentage_6_bar[0], percentage_6_bar[2], percentage_6_bar[4], percentage_6_bar[1], percentage_6_bar[3], percentage_6_bar[5]]
            Adjusted_header_plot = ['10 ms, 900 pps', '50 ms, 900 pps', '200 ms, 900 pps', '10 ms, 4500 pps', '50 ms, 4500 pps', '200 ms, 4500 pps']
            
            #modify the data bc of plotting reasons:
        
            paired_headers_6_bar = [f"{s}_{corhNum}" for s in modified_headers_6_bar[i_6_bar]]
            # print('mod_headers', modified_headers_6_bar)
            # print('percentages', Percentage_vals[i_6_bar])
            bars = ax_6_bar.bar(paired_headers_6_bar, modified_percentage_6_bar[i_6_bar], color = List_of_Cols[corhNum])
            for bar in bars:
                height = bar.get_height()  # Get bar height
                ax_6_bar.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', 
                        ha='center', va='bottom', fontweight='bold', fontsize=8)
                ax_6_bar.set_ylim(0,100)

            
            # ax_6_bar.xaxis.set_visible(False)  # Hides everything on the x-axis

            
            if i_6_bar == 0 or i_6_bar == 3:
                if contains_eve(str_recordingsites):
                    
                    ax_6_bar.set_ylabel('% Multi-units with VE>0')#, fontsize = 18)
                    # ax_6_bar.set_xticklabels(['EVEpt', 'EVEenv', 'EVEpt', 'EVEenv', 'EVEpt', 'EVEenv'], fontsize = 15)
                    # ax_6_bar.tick_params(axis = 'y')#, labelsize = 18)
                    # ax_6_bar.set_ylim(0,100)
                else:
                    ax_6_bar.set_ylabel('% Multi-units with VE>0')#, fontsize = 12)
                    # ax_6_bar.set_ylabel('Prozent signifikanter VE-Werte', fontsize = 18)
                    # ax_6_bar.set_xticklabels(['VE PT', 'VE ENV', 'VE PT', 'VE ENV', 'VE PT', 'VE ENV'], fontsize = 15)
                    # ax_6_bar.tick_params(axis = 'y', labelsize = 19)
                    # ax_6_bar.set_ylim(0,100)
                    
            if i_6_bar < 3:
                ax_6_bar.set_xticklabels([]) 
            else:
                if contains_eve(str_recordingsites):
                    ax_6_bar.set_xticklabels(['EVEpt', 'EVEenv', 'EVEpt', 'EVEenv', 'EVEpt', 'EVEenv'], rotation = 45)#, fontsize = 15)
                else:
                    ax_6_bar.set_xticklabels(['VE(PT)', 'VE(ENV)', 'VE(PT)', 'VE(ENV)', 'VE(PT)', 'VE(ENV)'], rotation = 45, fontstyle = 'italic', )
                    # ax_6_bar.set_xticklabels(['VEpt', 'VEenv', 'VEpt', 'VEenv', 'VEpt', 'VEenv'], fontsize = 15)


            
                
            plt.tight_layout()
            custom_legend = [
                Line2D([0], [0], color= AD_Acute_Color, lw=5, label='AD + acute CI'),  
                Line2D([0], [0], color= ND_Acute_Color, lw=5, label='ND + acute CI'), 
                Line2D([0], [0], color= ND_Chronic_Color, lw=5, label='ND + chronic CI') 
                ]
            # custom_legend = [
            #     Line2D([0], [0], color= AD_Acute_Color, lw=8, label='AD + akut CI'),  
            #     Line2D([0], [0], color= ND_Acute_Color, lw=8, label='ND + akut CI'), 
            #     Line2D([0], [0], color= ND_Chronic_Color, lw=8, label='ND + chronisch CI') 
            #     ]

            # Add the custom legend to the plot
            if corhNum == 2:
                # title = f"Combination: {paired_headers_6_bar[0][5:-2]}"
                # title = f"Combination: {Adjusted_header_plot[i_6_bar]}"
                title = f"{Adjusted_header_plot[i_6_bar]}"
                # title = f"Kombination: {Adjusted_header_plot[i_6_bar]}"
                ax_6_bar.set_title(title)#, fontsize = 19)
            # plt.subplots_adjust(bottom = 0.2)
            
            axes_6_bar[0,0].legend(handles=custom_legend, loc='upper right', fontsize = 8)
            # fig.supylabel('Percentage significant VE values')
            plt.show()
            # plt.savefig(figurepath + 'Comparison_Percentage_4.png', dpi = 300)

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


