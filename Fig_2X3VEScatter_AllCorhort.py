#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 14:02:48 2026

@author: shiyi
"""
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

# %%
'''
VE distribution plot
'''
analysisRoot = '/ephys/2/0_ENVvsPT/Analysis'
dataRoot = '/ephys/2/0_ENVvsPT/Data'

# dataFiles = pd.read_csv(analysisRoot+'/Salt_Beaker_CSV_Files/datafileInventory_Salt_Beaker_Agar.csv')
# baselinerecordingSites = pd.read_csv(analysisRoot+'/Salt_Beaker_CSV_Files/recordingsiteInventory_Salt_Beaker_Agar_baseline_test.csv')
# recordingSites = pd.read_csv(analysisRoot+'/Salt_Beaker_CSV_Files/recordingsiteInventory_Salt_Beaker_Agar.csv')
dataFiles = pd.read_csv(analysisRoot+'/datafileInventory.csv')
baselinerecordingSites = pd.read_csv('/ephys/2/0_ENVvsPT/Analysis/recordingsiteInventory_baseline.csv')
recordingSites = pd.read_csv('/ephys/2/0_ENVvsPT/Analysis/recordingsiteInventory.csv')
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

dataRoot = '/ephys/2/0_ENVvsPT/Data'
analysisRoot = '/ephys/2/0_ENVvsPT/Analysis'
figureRoot = '/ephys/2/0_ENVvsPT/Figure'

#%%
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
            f'{rate}pps{dur}s': {'PT': 0, 'ENV': 0, 'Total': [], 'Triplet': 0, 'Baseline': 0} 
            for rate in pps 
            for dur in durs
        }
        fig, axs = prepare2X3Plot()
        anims = dataFiles[dataFiles['condition']== corhort]['animDir'].unique()
        # print(anims)
        unitNum = []
        for animNum, anim in enumerate(anims):
            if anim == "2023_02_23":
                continue

            filenames = dataFiles[(dataFiles['animDir'] == anim) & (dataFiles['Cleaned'] == 1)]['fileName']
            # filenames = dataFiles[(dataFiles['animDir'] == anim)]
            print(anim)
            print(filenames)
            for penNum, filename in enumerate(filenames):
                # if penNum == 2:
                if True:
                    print(filename)
                    sub_baselinerecordingSites = baselinerecordingSites[(baselinerecordingSites['fileName'] == filename)]
                    sub_recordingSites = recordingSites[(recordingSites['fileName'] == filename)]
                    has_nan = sub_recordingSites.isna().any().any()
                    print(has_nan)
                    # unitNum += len(sub_recordingSites)
                    unitNum.append(len(sub_recordingSites))
                    scatterText = 'anim'+str(animNum)+'pen'+str(penNum)
                    fig = plot2X3Scatter(sub_recordingSites, sub_baselinerecordingSites, fig, axs, scatterText, type='significancy')
                    for ii in range(6):
                    # for ii in range(3, 4):
                        xx = ii % 3
                        yy = ii//3
                        rate = str(pps[yy])
                        dur = str(durs[xx])
                        key = f'{rate}pps{dur}s'
                        print(key)
                        VEheader, significanceHeader = Header(sub_recordingSites, ii)
                        
                        # check if any unit is removed by triplet, means NAN
                        nan_count = sub_recordingSites[significanceHeader[0]].isna().sum()
                        if nan_count != 0:
                            print(nan_count)
                        total_counts[key]['Triplet'] += nan_count
                        
                        # check the number of unit removed bc of baseline
                        remove_rows = sub_baselinerecordingSites[(sub_baselinerecordingSites[significanceHeader[0]] == 1) | (sub_baselinerecordingSites[significanceHeader[1]] == 1)]                     
                        total_counts[key]['Baseline'] += int(len(remove_rows))
                        
                        df = sub_baselinerecordingSites[(sub_baselinerecordingSites[significanceHeader[0]] == 0) & (sub_baselinerecordingSites[significanceHeader[1]] == 0)] # select out unit non significant to pt or env at baseline
                        keepidx = df.index
                        sub_subrecordingSites = sub_recordingSites.loc[keepidx]
                        # sub_subrecordingSites = sub_recordingSites
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
            ax.text(0.5, 0.95, f'significant pt#:{total_counts[key]["PT"]}\nsigniificant env#{total_counts[key]["ENV"]}\nremoved by triplet#{total_counts[key]["Triplet"]}\nremoved by baseline{total_counts[key]["Baseline"]}', transform=ax.transAxes, verticalalignment='top')
                        
                        # print(f'{rate}pps{dur}s: \n{ptNum}/32 is sensitive to PTITD;\n{envNum}/32 is sensitive to ENVITD')
        fig.text(0, 0.5, 'Variance explained by PT_ITD (%)',
                 va='center', rotation=-90, fontsize=16)
        fig.text(0.38, 0.053, 'Variance explained by ENV_ITD (%)',
                 va='center', fontsize=16)
        fig.suptitle(corhort+'_ResponsiveUnit#'+str(np.sum(unitNum)), fontsize=18)
            # plt.savefig('/ephys/2/0_ENVvsPT/Figure/Result_ScatterPlot/'+corhort