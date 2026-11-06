#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 20:48:05 2025

@author: shiyi
"""

# AnalysisPipeline_3 for ENV vs pt
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


'''
################################################################################################################################
This code is a individual code to analyse the AMUA results of the recorded cue trading data (Step 3)
################################################################################################################################
'''


#%%
'''
write down the important conditions before starting the pipeline- so adjust these parameters and tell the program which data is going to be analyzed. Call this variable: "Data_conds"
makes life maybe easier. 
Cue Trading: [('clickRate (Hz)',1, 100, 900]),[ ('ITD (ms)',[-120, -80, -40, 0, 40, 80, 120]), ('ILD',[-4, -1, 0, 1, 4])]
Env Vs Pt:  [('duration (s)',[0.01, 0.05, 0.2]), ('clickRate (Hz)',[900, 4500]), ('ITD (ms)',[-0.1, 0, 0.1]), ('env ITD (ms)',[-0.1, 0, 0.1])]
Weighting: ?



'''

# specify the stim params that were permuted to generate different stimulus conditions
stimVarParams = ['duration (s)', 'clickRate (Hz)', 'ITD (ms)', 'env ITD (ms)']


#%%

dataRoot = config.DATA_ROOT
analysisRoot = config.ANALYSIS_ROOT
figureRoot = config.FIGURE_ROOT

#%%
'''
Read in & write in
'''


def readEphysFileFromRow(dataset, row, dataRoot = dataRoot, loadOrig=False, loadAMUA=False, loadClean=False):
    filename = dataset.fileName.iloc[row]
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
            dataRoot+'/'+dataset.animDir.iloc[row]+'/'+filename)
        # swps, stimPar = checkSwpsSize(swps, stimPar)
        print(filename)
    # if exist 'Cleaned' column in csv file, only read in row labed as '1' in cleaned
    if any('Cleaned' in col for col in dataset.columns):
        if dataset.Cleaned.iloc[row] == 1:
            swps, stimPar = ep.readEphysFile(
                dataRoot+'/'+dataset.animDir.iloc[row]+'/'+filename)
            # swps, stimPar = checkSwpsSize(swps, stimPar)
        if dataset.Cleaned.iloc[row] == 0:
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
    filename = dataFiles['fileName'].iloc[row]
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
        ax.set_title(f'{float(dur)*1000:.0f} ms, {rate} pps')
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
    percentile_value = np.percentile(df_abs_std_baseline, 99)
    leftoverart = np.where(df_abs_std > percentile_value)[0]
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
    # percentile_value = np.percentile(df_abs_std_baseline, 95)
    percentile_value = np.percentile(df_abs_std_baseline, 99)
    leftoverart = np.where(df_abs_std > percentile_value)[0]
    # leftoverart = leftoverart + 1
    # leftoverart = leftoverart -1

    idxs = [item for item in idxs_prev if item not in leftoverart]
    idxs = np.array(idxs, dtype=int)

    return idxs


# %% VE calc
'''
variance explained calculation
read in datafileInventory.csv and recordingsiteInventory.csv
'''
# dataFiles = pd.read_csv(analysisRoot+'/Salt_Beaker_CSV_Files/datafileInventory_Salt_Beaker_Agar.csv')
# recordingSites = pd.read_csv(analysisRoot+'/Salt_Beaker_CSV_Files/recordingsiteInventory_Salt_Beaker_Agar.csv')

dataFiles=pd.read_csv(config.ANALYSIS_ROOT+'/datafileInventory.csv')
recordingSites = pd.read_csv(config.ANALYSIS_ROOT+'/recordingsiteInventory.csv')
# %% calcualte VE over stimulus window
# dataFiles = dataFiles[(dataFiles.condition == 'NH')].reset_index(drop=True)
dataRoot = config.FRESH_DIR
dataFiles = dataFiles[dataFiles.Cleaned == 1]
# dataFiles = dataFiles[dataFiles.fileName.str.contains('ICAC')]
for row in range(len(dataFiles)):
    # if dataFiles.loc[row, 'condition'] in ['ITD+']:
    # if row >= 33:
    # if row == 14:
    # if row in range(30, 31):
    if True:
        # if dataFiles.Cleaned[row] == 0:
        #     continue
        # if 'ICAC' in dataFiles.iloc[row].fileName:
        #     print(f"Skipping {dataFiles.iloc[row].fileName} as it contains 'ICAC'")
        #     continue
        fname = dataFiles.fileName.iloc[row]
        print(fname)
        if fname.endswith('.ephys'):
            fname = fname[:-6]
        csvName = fname+'_Clean.csv'
        animDir = dataFiles.animDir.iloc[row]
        stimPar_norm = ep.readEphysStimTable(config.DATA_ROOT+'/'+animDir+'/'+fname+'.csv')
        # Swps_norm, stimPar_norm = readEphysFileFromRow(dataFiles, row, dataRoot = config.DATA_ROOT, loadOrig = True)
        # if isinstance(Swps_norm, list): # comment by Jan: I don't know what this does. Seems suspect --> Answer from Tim: ths is to get rid of the data we considered "not clean" in the the datafileinventory by adding a 0 in the Cleaned column. 
        #     continue
        
        swpAMUA, stimParAMUA = readEphysFileFromRow(dataFiles, row, dataRoot = dataRoot, loadAMUA = True)
        # stimParCLEAN = pd.read_csv(config.DATA_ROOT+'/'+animDir+'/'+csvName)
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
            # if channel == 17:
                idx = sub_recordingSites.index[channel]
                # if sub_recordingSites.Responsive[idx]== 1:
                if True:
                    print ('Now analysing:', filename, 'channel:', channel)

                    remove_keys = LeftOverArtefact (nup, p, channel, stimParCLEAN)   #check here if some keys have to be removed -> keys in that case are durs and clickrates
                    idx = sub_recordingSites.index[channel]

                    CHECKING = []
                    exclude= None
                    # p = p[45:45+9]
                    for ii, stimulus in enumerate(p):                      
                        print (stimulus)
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

                        # # signal = poolChannel(swpAMUA[idxs+1], channel)[:int(stimulus[0]*swpAMUA[0].sampleRate), :].mean(axis = 0) #Warning: idxs = idxs -1 originally due to ep.sweepIndecesFor
                        # signal = stimParAMUA.iloc[idxs_prev][f'AMUA_Avg_Chan{channel}']

                        AMUAs = swpAMUA[idxs+1]
                        signal = []
                        for jj, amua in enumerate(AMUAs):
                            sig = amua.signal
                            # avg = np.mean(sig[int(duration*amua.sampleRate), chanNum])
                            avg = np.mean(sig[-50:, channel])
                            signal.append(avg)
                        if ii % 9 == 0:
                            # df = stimParAMUA.iloc[idxs_prev][['ITD (ms)', 'env ITD (ms)', f'AMUA_Avg_Chan{channel}']]
                            df = pd.DataFrame({'PT_ITD': stimulus[2].repeat(len(signal)), 'ENV_ITD': stimulus[3].repeat(len(signal)), 'AMUA': signal})
                            # df2 = pd.DataFrame({'PT_ITD': stimulus[2].repeat(len(signal)), 'ENV_ITD': stimulus[3].repeat(len(signal)), 'AMUA': signal})
                        else:
                            df = pd.concat([df,pd.DataFrame({'PT_ITD': stimulus[2].repeat(len(signal)), 'ENV_ITD': stimulus[3].repeat(len(signal)), 'AMUA': signal})], ignore_index=True)
                            # df2 = pd.concat([df2,pd.DataFrame({'PT_ITD': stimulus[2].repeat(len(signal)), 'ENV_ITD': stimulus[3].repeat(len(signal)), 'AMUA': signal})], ignore_index=True)
                            # df = pd.concat([df, stimParAMUA.iloc[idxs_prev][['ITD (ms)', 'env ITD (ms)', f'AMUA_Avg_Chan{channel}']]])
                        if ii % 9 == 8 and exclude == None:
                            # print(len(df1))
                            # df1.columns = ['PT_ITD', 'ENV_ITD', 'AMUA']
                            ss_pt, ss_env, ss_pt_p, ss_env_p = calcVarianceExplained(df)
                            # ss_pt2, ss_env2 = calcVarianceExplained(df2)
                            print(ss_pt, ss_env, ss_pt_p, ss_env_p)
                            # ss_pt_p, ss_env_p = permutationTestfor(df, 1000)
                            # print(ss_pt, ss_env, ss_pt_p1, ss_env_p1)
                            
                            recordingSites = checkKeyNamein(recordingSites, keyName)
                            recordingSites.at[idx, keyName+'_VEpt'] = ss_pt
                            recordingSites.at[idx, keyName+'_VEenv'] = ss_env
                            recordingSites.at[idx, '(pValue)'+keyName+'_VEpt'] = ss_pt_p
                            recordingSites.at[idx, '(pValue)'+keyName+'_VEenv'] = ss_env_p
                            if ss_pt_p < 0.05:
                                recordingSites.at[idx, keyName+'_VEpt_significance'] = 1
                            else:
                                recordingSites.at[idx, keyName+'_VEpt_significance'] = 0
                            if ss_env_p < 0.05:
                                recordingSites.at[idx, keyName+'_VEenv_significance'] = 1
                            else:
                                recordingSites.at[idx, keyName+'_VEenv_significance'] = 0

                            # recordingSites = writeValueinto(recordingSites, idx, ss_pt, ss_env, ss_pt_p, ss_env_p, keyName)
                                                       
# recordingSites.to_csv(analysisRoot+'/Debugging_csv_files/recordingsiteInventory_Debugging_baseline.csv', index=False)
recordingSites.to_csv(config.ANALYSIS_ROOT+'/recordingsiteInventory_baseline.csv', index=False)
# recordingSites.to_csv(analysisRoot+'/Salt_Beaker_CSV_Files/recordingsiteInventory_Salt_Beaker_Agar_baseline_test.csv', index=False)
# recordingSites.to_csv(analysisRoot+'/recordingsiteInventory_baseline_test.csv', index=False)

#% VE calculation over response window
# dataFiles = dataFiles[(dataFiles.condition == 'ITD+')].reset_index(drop=True)
dataRoot = config.FRESH_DIR
recordingSites = pd.read_csv(config.ANALYSIS_ROOT+'/recordingsiteInventory.csv')
# recordingSites = pd.read_csv(analysisRoot+'/Salt_Beaker_CSV_Files/recordingsiteInventory_Salt_Beaker_Agar.csv')
for row in range(len(dataFiles)):
    # if dataFiles.loc[row, 'condition'] in ['ITD+']:
    # if row >= 33:
    # if row == 1:
    # if row in range(29, 32):
    if True:
        # if dataFiles.Cleaned[row] == 0:
        #     continue
        # if 'ICAC' in dataFiles.iloc[row].fileName:
        #     print(f"Skipping {dataFiles.iloc[row].fileName} as it contains 'ICAC'")
        #     continue
        fname = dataFiles.fileName.iloc[row]
        
        if fname.endswith('.ephys'):
            fname = fname[:-6]
        csvName = fname+'_Clean.csv'
        animDir = dataFiles.animDir.iloc[row]
        stimPar_norm = ep.readEphysStimTable(config.DATA_ROOT+'/'+animDir+'/'+fname+'.csv')
        # Swps_norm, stimPar_norm = readEphysFileFromRow(dataFiles, row, loadOrig = True)
        # swpAMUA, stimParAMUA = readEphysFileFromRow(dataFiles, row, loadAMUA = True)
        # stimParCLEAN = pd.read_csv(dataRoot+'/'+animDir+'/'+csvName)
        # Swps_norm, stimPar_norm = readEphysFileFromRow(dataFiles, row, dataRoot = config.DATA_ROOT, loadOrig = True)
        # if isinstance(Swps_norm, list): # comment by Jan: I don't know what this does. Seems suspect --> Answer from Tim: ths is to get rid of the data we considered "not clean" in the the datafileinventory by adding a 0 in the Cleaned column. 
        #     continue
        
        swpAMUA, stimParAMUA = readEphysFileFromRow(dataFiles, row, dataRoot = dataRoot, loadAMUA = True)
        # stimParCLEAN = pd.read_csv(config.DATA_ROOT+'/'+animDir+'/'+csvName)
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
            # if channel == 17:
                idx = sub_recordingSites.index[channel]
                # if sub_recordingSites.Responsive[idx]== 1:
                if True:
                    print ('Now analysing:', filename, 'channel:', channel)

                    remove_keys = LeftOverArtefact (nup, p, channel, stimParCLEAN)   #check here if some keys have to be removed -> keys in that case are durs and clickrates
                    idx = sub_recordingSites.index[channel]

                    CHECKING = []
                    exclude= None
                    # p = p[45:45+9]
                    for ii, stimulus in enumerate(p):                      
                        print (stimulus)
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

                        # # signal = poolChannel(swpAMUA[idxs+1], channel)[:int(stimulus[0]*swpAMUA[0].sampleRate), :].mean(axis = 0) #Warning: idxs = idxs -1 originally due to ep.sweepIndecesFor
                        signal = stimParAMUA.iloc[idxs_prev][f'AMUA_Avg_Chan{channel}']

                        # AMUAs = swpAMUA[idxs+1]
                        # signal = []
                        # for jj, amua in enumerate(AMUAs):
                        #     sig = amua.signal
                        #     # avg = np.mean(sig[int(duration*amua.sampleRate), chanNum])
                        #     avg = np.mean(sig[-50:, channel])
                        #     signal.append(avg)
                        if ii % 9 == 0:
                            # df = stimParAMUA.iloc[idxs_prev][['ITD (ms)', 'env ITD (ms)', f'AMUA_Avg_Chan{channel}']]
                            df = pd.DataFrame({'PT_ITD': stimulus[2].repeat(len(signal)), 'ENV_ITD': stimulus[3].repeat(len(signal)), 'AMUA': signal})
                            # df2 = pd.DataFrame({'PT_ITD': stimulus[2].repeat(len(signal)), 'ENV_ITD': stimulus[3].repeat(len(signal)), 'AMUA': signal})
                        else:
                            df = pd.concat([df,pd.DataFrame({'PT_ITD': stimulus[2].repeat(len(signal)), 'ENV_ITD': stimulus[3].repeat(len(signal)), 'AMUA': signal})], ignore_index=True)
                            # df2 = pd.concat([df2,pd.DataFrame({'PT_ITD': stimulus[2].repeat(len(signal)), 'ENV_ITD': stimulus[3].repeat(len(signal)), 'AMUA': signal})], ignore_index=True)
                            # df = pd.concat([df, stimParAMUA.iloc[idxs_prev][['ITD (ms)', 'env ITD (ms)', f'AMUA_Avg_Chan{channel}']]])
                        if ii % 9 == 8 and exclude == None:
                            # print(len(df1))
                            # df1.columns = ['PT_ITD', 'ENV_ITD', 'AMUA']
                            ss_pt, ss_env, ss_pt_p, ss_env_p = calcVarianceExplained(df)
                            # ss_pt2, ss_env2 = calcVarianceExplained(df2)
                            print(ss_pt, ss_env, ss_pt_p, ss_env_p)
                            # ss_pt_p, ss_env_p = permutationTestfor(df, 1000)
                            # print(ss_pt, ss_env, ss_pt_p1, ss_env_p1)
                            
                            recordingSites = checkKeyNamein(recordingSites, keyName)
                            recordingSites.at[idx, keyName+'_VEpt'] = ss_pt
                            recordingSites.at[idx, keyName+'_VEenv'] = ss_env
                            recordingSites.at[idx, '(pValue)'+keyName+'_VEpt'] = ss_pt_p
                            recordingSites.at[idx, '(pValue)'+keyName+'_VEenv'] = ss_env_p
                            if ss_pt_p < 0.05:
                                recordingSites.at[idx, keyName+'_VEpt_significance'] = 1
                            else:
                                recordingSites.at[idx, keyName+'_VEpt_significance'] = 0
                            if ss_env_p < 0.05:
                                recordingSites.at[idx, keyName+'_VEenv_significance'] = 1
                            else:
                                recordingSites.at[idx, keyName+'_VEenv_significance'] = 0

                            # recordingSites = writeValueinto(recordingSites, idx, ss_pt, ss_env, ss_pt_p, ss_env_p, keyName)
#%                           
# recordingSites.to_csv(analysisRoot+'/Debugging_csv_files/recordingsiteInventory_Debugging_baseline.csv', index=False)
# recordingSites.to_csv(config.ANALYSIS_ROOT+'/recordingsiteInventory.csv', index=False)
recordingSites.to_csv(analysisRoot+'/recordingsiteInventory.csv', index=False)
# recordingSites.to_csv(analysisRoot+'/Salt_Beaker_CSV_Files/recordingsiteInventory_Salt_Beaker_Agar_test.csv', index=False)
#%%
# Check AMUA responses:
# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_V3.csv')
# recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventoryV3.csv')

#     # if dataFiles.loc[row, 'condition'] in ['ITD+']:
# row = 59
#     # if row in range(59, 60):
#     # if True:
# if dataFiles.Cleaned[row] == 0:
#     print('No Bueno')
# fname = dataFiles.fileName[row]
# csvName = fname[:-6]+'_Clean.csv'
# animDir = dataFiles.animDir[row]
# swpAMUA, stimParAMUA = readEphysFileFromRow(dataFiles, row, loadAMUA = True)
# stimParCLEAN = pd.read_csv(dataRoot+'/'+animDir+'/'+csvName)


# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_V3.csv')

# corhort = 'ND'
# animNum = 4
# penNum = 0
# chanNum = 15
# duration = 0.2
# clickRate = 900

# anim = dataFiles[(dataFiles['condition'] == corhort)]['animDir'].unique()[animNum]
# filename = dataFiles[(dataFiles['animDir'] == anim) & (dataFiles['Cleaned'] == 1)].iloc[penNum]['fileName']
# print(filename)
# swpsClean, stimParCLEAN = ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-6]+'_Clean')
# swpsAMUA, stimPar = ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-6]+'_AMUA')

# ITD=[-0.1, 0, 0.1]

# for itd_pt in ITD: 
#     for itd_env in ITD:
#         # print(itd_pt, itd_env)
#         conditions = [('duration (s)', duration), ('clickRate (Hz)', clickRate), ('ITD (ms)', itd_pt), ('env ITD (ms)', itd_env)]

#         plot32Channel_Sanity(swpsAMUA, conditions, stimParCLEAN, allRepeats=True, avgRepeats=False)
# signal = swpsAMUA[1], chanNum
# plt.plot(signal)

# plot3X3plot_Sanity(swpsAMUA, sub_conditions = [duration, clickRate])
# from datafilename find corresponding recording sites in recordingsiteInventory.csv
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

        conditions = [('duration (s)', duration), ('clickRate (Hz)', clickRate), ('ITD (ms)', pt_itd), ('env ITD (ms)', env_itd)]
        idxs_prev = findSweepsFor(conditions.copy(), stimPar)
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
# %%
'''
VE distribution plot
'''
analysisRoot = config.ANALYSIS_ROOT
dataRoot = config.DATA_ROOT

# dataFiles = pd.read_csv(analysisRoot+'/Salt_Beaker_CSV_Files/datafileInventory_Salt_Beaker_Agar.csv')
# baselinerecordingSites = pd.read_csv(analysisRoot+'/Salt_Beaker_CSV_Files/recordingsiteInventory_Salt_Beaker_Agar_baseline_test.csv')
# recordingSites = pd.read_csv(analysisRoot+'/Salt_Beaker_CSV_Files/recordingsiteInventory_Salt_Beaker_Agar.csv')
dataFiles = pd.read_csv(analysisRoot+'/datafileInventory.csv')
baselinerecordingSites = pd.read_csv(config.ANALYSIS_ROOT+'/recordingsiteInventory_baseline.csv')
recordingSites = pd.read_csv(config.ANALYSIS_ROOT+'/recordingsiteInventory.csv')
#%%
# pps = [900, 4500]
# durs = [0.01, 0.05, 0.2]
corhorts = dataFiles.condition.unique()
# total_counts = {
#     f'{rate}pps{dur}s': {'PT': 0, 'ENV': 0, 'Total': []} 
#     for rate in pps 
#     for dur in durs
# }
for corhNum, corhort in enumerate(corhorts):
    if True:
    # if corhNum == 2:        
        pps = [900, 4500]
        durs = [0.01, 0.05, 0.2]
        total_counts = {
            f'{rate}pps{dur}s': {'PT': 0, 'ENV': 0, 'Total': []} 
            for rate in pps 
            for dur in durs
        }
        fig, axs = prepare2X3Plot()
        anims = dataFiles[dataFiles['condition']== corhort]['animDir'].unique()
        print(anims)
        unitNum = []
        for animNum, anim in enumerate(anims):
            if anim == "2023_02_23":
                continue

            filenames = dataFiles[(dataFiles['animDir'] == anim) & (dataFiles['Cleaned'] == 1)]['fileName']
            for penNum, filename in enumerate(filenames):
                # if penNum == 2:
                if True:
                    print(filename)
                    sub_baselinerecordingSites = baselinerecordingSites[(baselinerecordingSites['fileName'] == filename)]
                    sub_recordingSites = recordingSites[(recordingSites['fileName'] == filename)]
                    # unitNum += len(sub_recordingSites)
                    unitNum.append(len(sub_recordingSites))
                    scatterText = 'anim'+str(animNum)+'pen'+str(penNum)
                    fig = plot2X3Scatter(sub_recordingSites, sub_baselinerecordingSites, fig, axs, scatterText, type='significancy')
                    for ii in range(6):
                        xx = ii % 3
                        yy = ii//3
                        rate = str(pps[yy])
                        dur = str(durs[xx])
                        key = f'{rate}pps{dur}s'
                        VEheader, significanceHeader = Header(sub_recordingSites, ii)
                        df = sub_baselinerecordingSites[(sub_baselinerecordingSites[significanceHeader[0]] == 0) & (sub_baselinerecordingSites[significanceHeader[1]] == 0)] # select out unit non significant to pt or env at baseline
                        keepidx = df.index
                        sub_subrecordingSites = sub_recordingSites.loc[keepidx]
                        sub_subrecordingSites = sub_recordingSites
                        ptNum = int(np.sum(sub_subrecordingSites[significanceHeader[0]]))
                        envNum = int(np.sum(sub_subrecordingSites[significanceHeader[1]]))
                        # df = sub_baselinerecordingSites[(sub_baselinerecordingSites[significanceHeader[0]] == 0) & (sub_baselinerecordingSites[significanceHeader[1]] == 0)]
                        
                        total_counts[key]['PT'] += ptNum
                        total_counts[key]['ENV'] += envNum
                        # total_counts[key]['Total'] += len(keepidx)
                        total_counts[key]['Total'].append(len(keepidx))
        for ii in range(6):                
            ax = axs[ii]
            xx = ii % 3
            yy = ii//3
            rate = str(pps[yy])
            dur = str(durs[xx])
            key = f'{rate}pps{dur}s'
            print(key)
            ax.set_title(f'{ax.get_title()} Total#: {np.sum(total_counts[key]["Total"]) }')
            ax.text(0.55, 0.95, f'significant pt#:{total_counts[key]["PT"]}\nsigniificant env#{total_counts[key]["ENV"]}', transform=ax.transAxes, verticalalignment='top')
                        
                        # print(f'{rate}pps{dur}s: \n{ptNum}/32 is sensitive to PTITD;\n{envNum}/32 is sensitive to ENVITD')
        fig.text(0, 0.5, 'Variance explained by PT_ITD (%)',
                 va='center', rotation=-90, fontsize=16)
        fig.text(0.38, 0.053, 'Variance explained by ENV_ITD (%)',
                 va='center', fontsize=16)
        fig.suptitle(corhort+'_ResponsiveUnit#'+str(np.sum(unitNum)), fontsize=18)
            # plt.savefig(config.FIGURE_ROOT+'/Result_ScatterPlot/'+corhort
# %%
'''
VE distribution plot, penetration label by different color
'''
import matplotlib.patches as mpatches
from matplotlib.cm import plasma
corhorts = dataFiles.condition.unique()

coloridx = 0
for corhNum, corhort in enumerate(corhorts):
    # if corhNum == 0:
    if True:
        fig, axs = prepare2X3Plot()
        anims = dataFiles[dataFiles['condition']== corhort]['animDir'].unique()
        print(anims)
        colors = plasma(np.linspace(0, 1, 10))
        np.random.shuffle(colors)
        unitNum = 0
        legend_elements = []
        for animNum, anim in enumerate(anims):
            filenames = dataFiles[(dataFiles['animDir'] == anim) & (dataFiles['Cleaned'] == 1)]['fileName']
            for penNum, filename in enumerate(filenames):
                if penNum == 2:
                # if True:
                    # sub_recordingSites = recordingSites[(recordingSites['fileName'] == filename) & (recordingSites['Responsive'] == 1)]
                    sub_baselinerecordingSites = baselinerecordingSites[(baselinerecordingSites['fileName'] == filename)]
                    sub_recordingSites = recordingSites[(recordingSites['fileName'] == filename)]
                    unitNum += len(sub_recordingSites)
                    scatterText = f'Animal {animNum+1} Pen {penNum+1}'
                    # fig = plot2X3Scatter(sub_recordingSites, fig, axs, scatterText, type = 'significancy')
                    coloridx += 1
                    color = colors[coloridx] 
                    fig = plot2X3Scatter(sub_recordingSites, sub_baselinerecordingSites, fig, axs, scatterText, type='penetration', colors=color)
                    legend_elements.append(mpatches.Patch(facecolor=color, 
                                                        edgecolor='black',
                                                        label=scatterText))                
        
        fig.legend(handles=legend_elements, 
          loc='center right',
          bbox_to_anchor=(0.98, 0.5))
        fig.text(0, 0.5, 'VE(PT) in %',
                 va='center', rotation=-90, fontsize=14)
        fig.text(0.38, 0.053, 'VE(ENV) in %',
                 va='center', fontsize=14)
        fig.suptitle(corhort+'_ResponsiveUnit#'+str(unitNum), fontsize=14)     
        # plt.savefig(config.FIGURE_ROOT+'/Result_ScatterPlot/'+corhort)
# %%
'''
plot out all channel(chs) which the VEpt exceed setting range
'''
checkParams = [4500.0, 0.01]

corhorts = dataFiles.condition.unique()
# colors = plt.cm.tab20(np.linspace(0, 1, 40))
for corhNum, corhort in enumerate(corhorts):
    if corhNum == 2:
        # if True:
        # fig, axs = prepare2X3Plot()
        # if not hasattr(fig, 'legend_elements'):
        #     fig.legend_elements = []
        anims = dataFiles[dataFiles['condition']== corhort]['animDir'].unique()
        print(anims)
        unitNum = 0
        # coloridx = 0
        for animNum, anim in enumerate(anims):
            # if animNum == 0:
            if True:
                # coloridx = 0
                # fig, axs = prepare2X3Plot()
                # if not hasattr(fig, 'legend_elements'):
                # fig.legend_elements = []
                filenames = dataFiles[(dataFiles['animDir'] == anim) & (dataFiles['Cleaned'] == 1)]['fileName']
                for penNum, filename in enumerate(filenames):
                    # if penNum == 0:
                    if True:
                        # if penNum in [0, 1]:
                        # sub_recordingSites = recordingSites[(recordingSites['fileName'] == filename) & (recordingSites['Responsive'] == 1)]
                        sub_recordingSites = baselinerecordingSites[baselinerecordingSites['fileName'] == filename]
                        key = f'Dur{str(checkParams[1])}_Rate{str(checkParams[0])}_VEpt'
                        i = np.where((sub_recordingSites[key] > 0.1) & (sub_recordingSites[f'{key}_significance'] == 1))[0]
                        targetidx = sub_recordingSites.index[i]
                        target = sub_recordingSites.iloc[i]
                        chs = target.Channel
                        if not chs.empty:
                            print(f'{filename},\ncorhNum:{corhNum},animNum:{animNum},penNum:{penNum}:\n{chs.values}')
# %%
'''
debug VE distribution
'''
# dataFiles = pd.read_csv(analysisRoot+'datafileInventory_Salt_Beaker_Agar.csv')
dataRoot = config.FRESH_DIR
corhort = 'ITD+'
animNum = 4
penNum = 0
chanNum = 16
duration = 0.05
clickRate = 4500

anim = dataFiles[(dataFiles['condition'] == corhort)]['animDir'].unique()[animNum]
filename = dataFiles[(dataFiles['animDir'] == anim) & (dataFiles['Cleaned'] == 1)].iloc[penNum]['fileName']
print(filename)
swpsClean, stimParCLEAN = ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-5]+'_Clean')
swpsAMUA, stimPar = ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-5]+'_AMUA')
# swps, stimPar = ep.readEphysFile(dataRoot+'/'+anim+'/'+filename)
# %% plot clean data

for chanNum in range(32):
# for chanNum in chs:
    if True:
    # if chanNum == 0:
        # sampleRate = swpsClean[1].sampleRate
        sampleRate = swpsAMUA[1].sampleRate
        t = np.arange(0, duration+0.05, 1/sampleRate)
        
        mask = (recordingSites['fileName'] == filename) & (recordingSites['Channel'] == chanNum)
        keypt = f'Dur{duration}_Rate{clickRate}.0_VEpt'
        keyenv = f'Dur{duration}_Rate{clickRate}.0_VEenv'
        VEpt = recordingSites.loc[mask, keypt].values[0]
        VEenv = recordingSites.loc[mask, keyenv].values[0]
        print([VEpt, VEenv])
        
        fig, axs = prepare3X3Plot()
    
        itd = [-0.1, 0, 0.1]
        for ii in range(9):
            xx = ii % 3
            yy = ii//3
            pt_itd = itd[yy]
            env_itd = itd[xx]
            ax = axs[ii]
            # print(ax.title, pt_itd, env_itd)
            # conditions=[('duration (s)',duration), ('clickRate (Hz)',clickRate), ('ITD (ms)',pt_itd), ('env ITD (ms)',env_itd)]
            # idxs_prev1 =findSweepsFor(conditions.copy(), stimPar)
            keys = ['duration (s)', 'clickRate (Hz)', 'ITD (ms)', 'env ITD (ms)']
            conds = np.array([duration, clickRate, pt_itd, env_itd])
            print(ii, conds)
            idxs_prev = ep.stimIndecesFor(stimParCLEAN, keys, conds)
    
            # swpIdx = remove_single_ARTsweeps(idxs_prev, chanNum, stimParCLEAN)
            # swpIdx = swpIdx + 1
            swpIdx = idxs_prev+1
            
            # because when calculating VE baseline, time window is 50samples from end
            # below to check the baseline as each swp has differenct length, cannot use poolchannel directly
            ''''''
            # sigs = swpsClean[swpIdx]
            sigs = swpsAMUA[swpIdx]
            signals = []
            for idx, sig in enumerate(sigs):
                # signal = sig.signal[-50:, chanNum]
                signal = sig.signal[-500:, chanNum]
                signals.append(signal)
                # print(signal.shape)
                # ax.plot(signal)
                ax.plot(signal[::-1])
            signals = np.mean(np.array(signals), axis = 0)
            ax.plot(signals[::-1], 'r')
            ax.axvline(x = 50, color='k', linestyle='--')
            ''''''
            # signal = poolChannel(swpsClean[swpIdx], chanNum)
            # # ax.plot(t, np.mean(signal[:len(t)], axis = 1))
            # # signal = signal[:len(t)]
            # ax.plot(t, signal)
            # # ax.plot(t, np.mean(signal, axis = 1), color = 'r')
            # # ax.plot(signal)
            # ax.plot(np.mean(signal, axis = 1), color = 'r')
            # ax.axvline(x=int(duration*sampleRate), color='k', linestyle='--')
            # # ax.axhline(y = np.mean(signal), color = 'k', linestyle = '--')
            # # ax.axvline(x=duration, color='k', linestyle='--')
            ax.set_title(f'{ax.title._text}, swpNo:{len(swpIdx)}')
            ax.set_ylim(-0.2e-5, 2.5e-5)
            # ax.set_ylim(-0.2e-5, 1.5e-5)
        fig.suptitle(f'{filename}, ch{chanNum}, VEpt{VEpt*100:.2f}, VEenv{VEenv*100:.2f}')
        
#%% check avg AMUA distribution
# check 4500pps, 0.2s 
# swpsAMUA, stimPar = ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-5]+'_AMUA')
# chanNum = 4
#%%
avgName = f'AMUA_Avg_Chan{chanNum}'
print(avgName)
fig, axs = prepare3X3Plot()
mask = (recordingSites['fileName'] == filename) & (recordingSites['Channel'] == chanNum)
keypt = f'Dur{duration}_Rate{clickRate}.0_VEpt'
keyenv = f'Dur{duration}_Rate{clickRate}.0_VEenv'
VEpt = recordingSites.loc[mask, keypt].values[0]
VEenv = recordingSites.loc[mask, keyenv].values[0]
title = f'{filename}, ch{chanNum}, VEpt{VEpt*100:.2f}, VEenv{VEenv*100:.2f}'
fig.suptitle(title)
for ii in range(9):
    xx = ii % 3
    yy = ii//3
    pt_itd = itd[yy]
    env_itd = itd[xx]
    ax = axs[ii]
    
    conditions=[('duration (s)',duration), ('clickRate (Hz)',clickRate), ('ITD (ms)',pt_itd), ('env ITD (ms)',env_itd)]
    # conditions=[('duration (s)',duration), ('clickRate (Hz)',clickRate)]
    print(conditions)
    idxs_prev1 =findSweepsFor(conditions.copy(), stimPar) # this is swpidx. for stimidx need to -1
    
    # from swpsAMUA calculate avgAMUA
    AMUAs = swpsAMUA[idxs_prev1]
    avgs = []
    for ii, amua in enumerate(AMUAs):
        sig = amua.signal
        # avg = np.mean(sig[int(duration*amua.sampleRate), chanNum])
        avg = np.mean(sig[-50:, chanNum])
        avgs.append(avg)
        
    avg = np.mean(avgs)
    ax.hist(avgs, bins = 20, label=f'avgAMUA{avg:.5e}')
    ax.axvline(x = avg, color = 'r', linestyle = '--')
    
    # from swpsClean calculate rmsClean
    # Cleans = swpsClean[idxs_prev1]
    # rmss = []
    # for ii, clean in enumerate(Cleans):
    #     sig = clean.signal
    #     # rms = np.sqrt(np.mean(sig[:int(duration*clean.sampleRate), chanNum]**2))
    #     rms = np.sqrt(np.mean(sig[-4000:, chanNum]**2))
    #     rmss.append(rms)
        
    # avg = np.mean(rmss)
    # ax.hist(rmss, bins = 20, label=f'avgRMS{avg:.5e}')
    # ax.axvline(x = avg, color = 'r', linestyle = '--')
         
    
    # from stimAMUA get avgAMUA
    # stimIdx = idxs_prev1-1
    # avgAMUA = stimPar.loc[stimIdx, avgName]
    # avg = np.mean(avgAMUA)
    # ax.hist(avgAMUA, bins = 20, label=f'avgAMUA{avg:.2e}')   
    # ax.axvline(x = avg, color = 'r', linestyle = '--')  
    
    ax.set_ylim(0, 5)
    # ax.set_xlim(4e-6, 5.8e-6) 
    ax.legend()
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
figurepath = config.FIGURE_ROOT+'/'
figurepath = config.FIGURE_ROOT+'/'

analysisRoot = config.ANALYSIS_ROOT
dataRoot = config.DATA_ROOT
dataFiles = pd.read_csv(analysisRoot+'/datafileInventory.csv')
# dataFiles = pd.read_csv(analysisRoot+'/Salt_Beaker_CSV_Files/datafileInventory_Salt_Beaker_Agar.csv')
# baselinerecordingSites = pd.read_csv(analysisRoot+'/Salt_Beaker_CSV_Files/recordingsiteInventory_Salt_Beaker_Agar_baseline.csv')
# recordingSites = pd.read_csv(analysisRoot+'/Salt_Beaker_CSV_Files/recordingsiteInventory_Salt_Beaker_Agar.csv')
baselinerecordingSites = pd.read_csv(config.ANALYSIS_ROOT+'/recordingsiteInventory_baseline.csv')
recordingSites = pd.read_csv(config.ANALYSIS_ROOT+'/recordingsiteInventory.csv')

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
    filenames = dataFiles[(dataFiles['animDir'] == anim) & (dataFiles['Cleaned'] == 1)]['fileName']
    if len(filenames) == 0:
        continue
    sub_recordingSites = []
    for filename in filenames:
        sub_recordingSites.append(
            recordingSites[recordingSites['fileName'] == filename])
    sub_recordingSites = pd.concat(sub_recordingSites, ignore_index=True)
    # sub_recordingSites = sub_recordingSites[sub_recordingSites['Responsive'] == 1]
    Anims.append(sub_recordingSites)

Anims_baseline = []
for anim in animDir:
    filenames = dataFiles[(dataFiles['animDir'] == anim)& (dataFiles['Cleaned'] == 1)]['fileName']
    # filename = dataFiles[(dataFiles['animDir'] == anim) & (dataFiles['Cleaned'] == 1)].iloc[penNum]['fileName']
    if len(filenames) == 0:
        continue
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
    
    # Horizontal
    fig_box, axes_box = plt.subplots(1, 3, figsize=(8.1, 6), sharex=True)
    
    fig_box.subplots_adjust(right=0.98, wspace=0.12)
    highlighted_groups = [(3, 4), (7, 8), (11, 12)]
    flierprops = dict(marker='o', markersize=0.2, color='black', alpha=0.7)
    
    for i, ax in enumerate(axes_box):
        ax.set_xlim(-5, 100)
        ax.grid(False)  # REMOVE GRID HERE
        
        for start, end in highlighted_groups:
            ax.add_patch(
                patches.Rectangle(
                    (-5, start - 0.5),
                    105, 
                    end - start + 1, 
                    facecolor="palegoldenrod",
                    alpha=0.25,
                    zorder=0
                )
            )    
    
    # Create other figures
    fig_bar, axes_bar = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    for ax in axes_bar.flat if hasattr(axes_bar, 'flat') else [axes_bar]:
        ax.grid(False)  # REMOVE GRID HERE
    
    fig_6_bar, axes_6_bar = plt.subplots(2, 3, figsize=(7.2, 6), sharey=True)
    for ax in axes_6_bar.flat:
        ax.grid(False)  # REMOVE GRID HERE
    
    if contains_eve(str_recordingsites):
        fig_hist, axes_hist = plt.subplots(3, 6, figsize=(18, 10), sharex=True)
        for ax in axes_hist.flat:
            ax.grid(False)  # REMOVE GRID HERE


All_VE_Data_Outside = []
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
        # if idx == 7:# or idx == 19 or idx == 13:
        #     continue
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
        # if idx_base == 7:# or idx == 19 or idx == 13:
        #     continue
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
        All_VE_Data_Outside.append(Combined_Data)

    # Combined_Data = Combined_Data.dropna(axis=1, how='all')
    # Combined_Data = Combined_Data.fillna(1)
    # No_All_Units = np.sum(all_units)
    print(No_All_Units)
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
                plt.savefig(figurepath + 'Comparison_Boxplot_NewFig.6.png', dpi = 300)

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
                    
                    title = f"{corhort}"
                    ax_box.set_title(title)
                    # ax_box.set_ylim(-5, 100)
        
                    ax_box.set_xlabel('VE values in %', fontsize = 12)
                        
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
            # Add p-values to each subplot
        
        
        # Define p-values for each of the 3 subplots (6 p-values per subplot)
        p_vals = ['p < 0.001', 'p < 0.001', 'p < 0.001', 'p < 0.001', 'p < 0.001', 'p < 0.001', 'p < 0.001', 'p = 0.000163', 'p = 0.00172', 'p < 0.001', 'p = 0.00336', 'p = 0.000275', 'p < 0.001', 'p < 0.001', 'p < 0.001', 'p < 0.001', 'p < 0.001', 'p < 0.001']
        
        #p-values extracted from last cell of MEM_analysis.py
        p_val_sets = [
            p_vals[0:6],   # First 6 p-values for left subplot
            p_vals[6:12],  # Next 6 p-values for middle subplot  
            p_vals[12:18]  # Last 6 p-values for right subplot
        ]
        
        # For each subplot, add the p-values
        for i_box, ax_box in enumerate(axes_box):
            # Get the p-values for this subplot
            current_p_vals = p_val_sets[i_box]
            
            # Define y positions for this subplot (midpoints between boxplot pairs)
            # These will depend on which boxplots are being compared
            if i_box == 0:
                # Left subplot: positions between boxplots 1-2, 3-4, 5-6, 7-8, 9-10, 11-12
                y_positions = [2, 4, 6, 8, 10, 12]
            elif i_box == 1:
                # Middle subplot: positions between boxplots 2-3, 4-5, 6-7, 8-9, 10-11, 12-1(next?)
                # Adjust based on your comparison logic
                y_positions = [2, 4, 6, 8, 10, 12]
            else:  # i_box == 2
                # Right subplot
                y_positions = [2, 4, 6, 8, 10, 12]
            
            # Add each p-value text
            for j, p_val in enumerate(current_p_vals):
                x_pos = ax_box.get_xlim()[1] * 0.97  # 97% of the way to the right edge
                y_pos = y_positions[j]
                
                # Add the text
                ax_box.text(x_pos, y_pos, p_val, 
                           ha='right', va='center',
                           fontsize=9,
                           fontweight='bold',  # Make p-values stand out
                           zorder=2)
                

                # plt.subplots_adjust(bottom = 0.2)
                # plt.show()
                # bplot.savefig(figurepath + 'Comparison_Boxplot_NewFig.5_Horizontally.png', dpi = 300)



        ##################################### Barplot with each cohort seperately in a subplot (3,1) ####################################
        # plt.figure()
        # p-vals (from last cell in MEM_analysis.py)
        #Starting from NH - 4500- 0.2,0.05,0.01 to NH -900- 0.2,0.05,0.01, to ND - 4500- 0.2,0.05,0.01...
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
                    
                    ax_6_bar.set_ylabel('Multi-units with VE>0 in %')#, fontsize = 18)
                    # ax_6_bar.set_xticklabels(['EVEpt', 'EVEenv', 'EVEpt', 'EVEenv', 'EVEpt', 'EVEenv'], fontsize = 15)
                    # ax_6_bar.tick_params(axis = 'y')#, labelsize = 18)
                    # ax_6_bar.set_ylim(0,100)
                else:
                    ax_6_bar.set_ylabel('Multi-units with VE>0 in %')#, fontsize = 12)
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
            # plt.savefig(figurepath + 'Comparison_Percentage_7.png', dpi = 300)

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



