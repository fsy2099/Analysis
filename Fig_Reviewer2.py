#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 15:10:08 2026

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
    gs.update(left=0.08, right=0.98, bottom=0.12,
              top=0.92, wspace=0.15, hspace=0.18)
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
                # sub_subrecordingSites = sub_recordingSites.loc[keepidx]
                sub_subrecordingSites = sub_recordingSites
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
'''
VE distribution plot
'''
analysisRoot = '/ephys/2/0_ENVvsPT/Analysis'
dataRoot = '/ephys/2/0_ENVvsPT/Data'

dataFiles = pd.read_csv(analysisRoot+'/Salt_Beaker_CSV_Files/datafileInventory_Salt_Beaker_Agar.csv')
baselinerecordingSites = pd.read_csv(analysisRoot+'/Salt_Beaker_CSV_Files/recordingsiteInventory_Salt_Beaker_Agar_baseline_test.csv')
recordingSites = pd.read_csv(analysisRoot+'/Salt_Beaker_CSV_Files/recordingsiteInventory_Salt_Beaker_Agar.csv')
#%%
'''
VE distribution plot, penetration label by different color
'''
import matplotlib.patches as mpatches
from matplotlib.cm import plasma
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
figureRoot = '/ephys/2/0_ENVvsPT/Figure/Appendix'
durs = [0.01, 0.05, 0.2]  # duration in seconds
pps = [900, 4500]         # pulses per second

def prepare2X3Plot(durs=[0.01, 0.05, 0.2], pps=[900, 4500]):
    fig = plt.figure(figsize=(12, 8))
    
    # 固定的GridSpec布局参数
    gs = fig.add_gridspec(2, 3)
    gs.update(left=0.08,        # 固定左边距
             right=0.95,
             bottom=0.08,       # 固定下边距
             top=0.95,
             wspace=0.1,
             hspace=0.1)
    
    axs = []
    for ii in range(6):
        xx = ii % 3
        yy = ii//3
        rate = str(pps[yy])
        dur = str(durs[xx])
        ax = fig.add_subplot(gs[yy, xx])
        ax.set_title(f'{float(dur)*1000:.0f} ms, {rate} pps')
        axs.append(ax)
    return fig, axs

corhorts = dataFiles.condition.unique()
coloridx = 0

for corhNum, corhort in enumerate(corhorts):
    if True:
        fig, axs = prepare2X3Plot()
        anims = dataFiles[dataFiles['condition']== corhort]['animDir'].unique()
        print(anims)
        colors = plasma(np.linspace(0, 1, 10))
        unitNum = 0
        legend_elements = []
        
        # 遍历并绘制数据
        for animNum, anim in enumerate(anims):
            filenames = dataFiles[(dataFiles['animDir'] == anim) & (dataFiles['Cleaned'] == 1)]['fileName']
            for penNum, filename in enumerate(filenames):
                if penNum == 2:
                    sub_baselinerecordingSites = baselinerecordingSites[(baselinerecordingSites['fileName'] == filename)]
                    sub_recordingSites = recordingSites[(recordingSites['fileName'] == filename)]
                    unitNum += len(sub_recordingSites)
                    scatterText = f'Animal {animNum+1} Pen {penNum+1}'
                    coloridx += 1
                    color = colors[coloridx] 
                    fig = plot2X3Scatter(sub_recordingSites, sub_baselinerecordingSites, fig, axs, scatterText, type='penetration', colors=color)
                    legend_elements.append(mpatches.Patch(facecolor=color, 
                                                        edgecolor='black',
                                                        label=scatterText))
        
        # 调整坐标轴显示
        for i, ax in enumerate(axs):
            ax.set_xlim(-5, 100)
            ax.set_ylim(-5, 100)
            ax.tick_params(axis='both', labelsize=12)
            
            if i < 3:  # 非最下行
                ax.set_xticklabels([])
            
            if i % 3 != 0:  # 非最左列
                ax.set_yticklabels([])

        # 调整坐标轴标签位置，使其更靠近子图
        fig.text(0.02,                  # 更靠近y轴
                0.5,
                'VE(PT) in %',
                va='center',
                ha='center',
                rotation=90,
                fontsize=14)
        
        fig.text(0.5,
                0.02,                   # 更靠近x轴
                'VE(ENV) in %',
                va='center',
                ha='center',
                fontsize=14)

        # 为每个子图创建单独的放大图
        for i, ax in enumerate(axs):
            # 获取当前子图的数据
            xx = i % 3
            yy = i//3
            rate = str(pps[yy])
            dur = str(durs[xx])
            
            # 创建新的图形
            fig_zoom = plt.figure(figsize=(6, 6))
            ax_zoom = fig_zoom.add_subplot(111)
            
            # 复制原子图的数据到新图
            for line in ax.get_children():
                if isinstance(line, matplotlib.collections.PathCollection):  # 对于散点图
                    new_scatter = ax_zoom.scatter(line.get_offsets()[:, 0], 
                                                line.get_offsets()[:, 1],
                                                c=line.get_facecolors(),
                                                edgecolors=line.get_edgecolors(),
                                                alpha=line.get_alpha())
            
            # 设置新图的属性
            ax_zoom.set_xlim(-4, 10)
            ax_zoom.set_ylim(-4, 10)
            ax_zoom.set_xticks([0, 5, 10])  # 设置x轴刻度
            ax_zoom.set_yticks([0, 5, 10])  # 设置y轴刻度
            ax_zoom.tick_params(axis='both', labelsize=12)
            
            # 调整布局
            fig_zoom.tight_layout()
            
            save_path = f'{figureRoot}/zoom_plot_{corhort}_{dur}ms_{rate}pps.png'
            fig_zoom.savefig(save_path, dpi=300, bbox_inches='tight')
            
        save_path = f'{figureRoot}/main_plot_{corhort}.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')    
# %%
# '''
# example of original data, template and clean data
# '''
# corhort = 'SB'
# animNum = 0
# penNum = 2
# chanNum = 16
# duration = 0.05
# clickRate = 4500

# anim = dataFiles[(dataFiles['condition'] == corhort)]['animDir'].unique()[animNum]
# filename = dataFiles[(dataFiles['animDir'] == anim) & (dataFiles['Cleaned'] == 1)].iloc[penNum]['fileName']
# print(f'{anim}, {filename}')
# swpsClean, stimParCLEAN = ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-5]+'_Clean')
# swpsAMUA, stimPar = ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-5]+'_AMUA')
# swps, stimPar = ep.readEphysFile(dataRoot+'/'+anim+'/'+filename)
# # %% plot clean data
# sampleRate = swpsClean[1].sampleRate
# # sampleRate = swpsAMUA[1].sampleRate
# t = np.arange(0, duration+0.05, 1/sampleRate)

# mask = (recordingSites['fileName'] == filename) & (recordingSites['Channel'] == chanNum)
# keypt = f'Dur{duration}_Rate{clickRate}.0_VEpt'
# keyenv = f'Dur{duration}_Rate{clickRate}.0_VEenv'
# VEpt = recordingSites.loc[mask, keypt].values[0]
# VEenv = recordingSites.loc[mask, keyenv].values[0]
# print([VEpt, VEenv])

# fig, axs = prepare3X3Plot()

# itd = [-0.1, 0, 0.1]
# for ii in range(9):
#     xx = ii % 3
#     yy = ii//3
#     pt_itd = itd[yy]
#     env_itd = itd[xx]
#     ax = axs[ii]
#     # print(ax.title, pt_itd, env_itd)
#     # conditions=[('duration (s)',duration), ('clickRate (Hz)',clickRate), ('ITD (ms)',pt_itd), ('env ITD (ms)',env_itd)]
#     # idxs_prev1 =findSweepsFor(conditions.copy(), stimPar)
#     keys = ['duration (s)', 'clickRate (Hz)', 'ITD (ms)', 'env ITD (ms)']
#     conds = np.array([duration, clickRate, pt_itd, env_itd])
#     print(ii, conds)
#     idxs_prev = ep.stimIndecesFor(stimParCLEAN, keys, conds)

#     # swpIdx = remove_single_ARTsweeps(idxs_prev, chanNum, stimParCLEAN)
#     # swpIdx = swpIdx + 1
#     swpIdx = idxs_prev+1
    
#     # because when calculating VE baseline, time window is 50samples from end
#     # below to check the baseline as each swp has differenct length, cannot use poolchannel directly
#     ''''''
#     # sigs = swpsClean[swpIdx]
#     # # sigs = swpsAMUA[swpIdx]
#     # signals = []
#     # for idx, sig in enumerate(sigs):
#     #     # signal = sig.signal[-50:, chanNum]
#     #     # signal = sig.signal[-500:, chanNum]
#     #     sigal = sig.signal[:len(t), chanNum]
#     #     signals.append(signal)
#     #     # print(signal.shape)
#     #     ax.plot(t, signal)
#     #     # ax.plot(signal[::-1])
#     # signals = np.mean(np.array(signals), axis = 0)
#     # ax.plot(signals, 'r')
#     # ax.axvline(x = 50, color='k', linestyle='--')
#     ''''''
#     signal = poolChannel(swpsClean[swpIdx], chanNum)
#     # ax.plot(t, np.mean(signal[:len(t)], axis = 1))
#     signal = signal[:len(t)]
#     ax.plot(t, signal)
#     # ax.plot(t, np.mean(signal, axis = 1), color = 'r')
#     # ax.plot(signal)
#     # ax.plot(np.mean(signal, axis = 1), color = 'r')
#     # ax.axvline(x=int(duration*sampleRate), color='k', linestyle='--')
#     # ax.axhline(y = np.mean(signal), color = 'k', linestyle = '--')
#     ax.axvline(x=duration, color='k', linestyle='--')
#     ax.set_title(f'{ax.title._text}, swpNo:{len(swpIdx)}')
#     # ax.set_ylim(-0.2e-5, 2.5e-5)
#     # ax.set_ylim(-0.2e-5, 1.5e-5)
# fig.suptitle(f'{filename}, ch{chanNum}, VEpt{VEpt*100:.2f}, VEenv{VEenv*100:.2f}')
# #%%
