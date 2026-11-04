#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:53:30 2023

@author: shiyi
"""
import os 
import numpy as np
from scipy.signal import butter, filtfilt
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns 
from scipy import stats
#from matplotlib_scalebar.scalebar import ScaleBar
import pandas as pd
from cycler import cycler
from matplotlib.lines import Line2D
#Sig_path = '/disks/CIdata/1_ENVvsPT/2_ND/2022_01_24/Results_Data/'
#file_names = os.listdir(Sig_path)
#sig_names = [file_name for file_name in file_names if all([x in file_name for x in [ 'P5', "_OriginSigArray_AMUA.npy"]])]
#sig_name = sig_names[0]
#ipsi_amua = np.load(Sig_path+sig_name)
    

#%%
VE_path = '/ephys/2/CIproject/1_ENVvsPT/ANOVA/ND/'

Sig_path = '/ephys/2/CIproject/1_ENVvsPT/2_ND/2022_01_25/Results_Data/'
file_names = os.listdir(Sig_path)
sig_names = [file_name for file_name in file_names if all([x in file_name for x in [ 'P7', "_OriginSigArray_AMUA.npy"]])]
sig_name = sig_names[0]
contra_amua = np.load(Sig_path+sig_name)
contraVE_name = sig_name[:-4]+'_ExplainedVariance.npy'
contra_VE = np.load(VE_path+contraVE_name)

sig_names = [file_name for file_name in file_names if all([x in file_name for x in [ 'P3', "_OriginSigArray_AMUA.npy"]])]
sig_name = sig_names[0]
V_amua = np.load(Sig_path+sig_name)
VVE_name = sig_name[:-4]+'_ExplainedVariance.npy'
V_VE = np.load(VE_path+VVE_name)

Fs_down = 2000
Wn = 2*200/Fs_down
bLow,aLow = butter(2, Wn, 'lowpass')
t_start = 0.003
t_end = 0.1
t_end_heatmap = 0.053
t = np.arange(t_start, t_end, 1/Fs_down)*1000
line_xaxis = np.arange(9)
stiDur = [0.01, 0.05, 0.2]
stiITD = [-0.1, 0, 0.1]
channel = [10, 19]
ff = 0
dd = 1
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
#%%
fig1 = plt.figure(figsize=(12, 8),constrained_layout=True)
# AMUA trace
# gs0 is the grid filed for the background of AMUA trace
gs0 = gridspec.GridSpec(1, 1)
gs0.update(left=0.025, right=0.99, bottom = 0.05, top = 0.95, wspace=0, hspace=0)
# gs1 is the grid filed for #1 AMUA trace
gs1 = gridspec.GridSpec(2, 3)
gs1.update(left=0.09-0.02, right=0.85, bottom = 0.10, top = 0.50, wspace=0, hspace=0)
# gs3 is the grid filed for #1 AMUA trace
gs3 = gridspec.GridSpec(2, 3)
gs3.update(left=0.09-0.02, right=0.85, bottom = 0.55, top = 0.95, wspace=0, hspace=0)
#gs3 = gridspec.GridSpec(1, 1)
#gs3.update(left=0.07-0.03, right=0.97-0.03, bottom = 0.05, top = 0.95, wspace=0, hspace=0)
# AMUA heatmap
#gs2 = gridspec.GridSpec(2, 1)
#gs2.update(left=0.75-0.03, right=0.98-0.03, bottom = 0.13, top = 0.93, wspace=0, hspace=0.2)

ax0 = fig1.add_subplot(gs0[0, 0])
ax0_xticks = [0.17, 0.48, 0.79]
ax0_yticks = [0.2, 0.4, 0.65, 0.85]
#ax0_ylabels = ('ND-Unit#2', 'ND-Unit#1')
#ax0_ylabels = ('ENV_ITD', 'PT_ITD', 'ENV_ITD', 'PT_ITD')
#ax0_xlabels = ('-0.1', '0', '0.1')
ax0.set_ylabel('Normalized AMUA', rotation =  'vertical', fontsize = 11)
ax0.set_xlabel('Peristimulus time(ms)', fontsize = 11)
#ax0.set_xticks(ax0_xticks)
ax0.set_xticks([])
ax0.set_yticks([])
#ax0.set_xticklabels(ax0_xlabels)
#ax0.set_yticklabels(ax0_ylabels, rotation='vertical',  va="center")
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['left'].set_visible(False)
ax0.spines['bottom'].set_visible(False)
ax0.text(-0.02, 1.01, 'A', fontsize = 14)
#%%
# set up background color for AMUA trace
#l, b, w, h = ax0.get_position().bounds
#ax_background = fig.add_axes([l+0.02, b+0.69, 0.63, 0.205])
#ax_background.set_facecolor('gainsboro')
#ax_background.set_xticks([])
#ax_background.set_yticks([])
#ax_background.spines['top'].set_visible(False)
#ax_background.spines['right'].set_visible(False)
#ax_background.spines['left'].set_visible(False)
#ax_background.spines['bottom'].set_visible(False)
#ax_background = fig.add_axes([l+0.02, b+0.27, 0.63, 0.205])
#ax_background.set_facecolor('gainsboro')
#ax_background.set_xticks([])
#ax_background.set_yticks([])
#ax_background.spines['top'].set_visible(False)
#ax_background.spines['right'].set_visible(False)
#ax_background.spines['left'].set_visible(False)
#ax_background.spines['bottom'].set_visible(False)
#ax1 = fig.add_subplot(gs3[0, 0])

#ax1.set_xlim([0,1])
#ax1.set_ylim([0,1])
#ax1.set_xticks([])
#ax1.set_yticks([])
#ax1.spines['top'].set_visible(False)
#ax1.spines['right'].set_visible(False)
#ax1.spines['left'].set_visible(False)
#ax1.spines['bottom'].set_visible(False)

#ax1.arrow(0.63, 0.75, 0.1, 0, width = 0.003, fc='k', ec='k')
#ax1.arrow(0.63, 0.3, 0.1, 0, width = 0.003, fc='k', ec='k')
#%%
env_label = ['ENV_ITD -0.1', 'ENV_ITD 0', 'ENV_ITD 0.1']
pt_label = ['PT_ITD -0.1', 'PT_ITD 0', 'PT_ITD 0.1']
dot = np.arange(-0.2, 1.2, step = 0.2)
dot_x = np.repeat(t_end_heatmap*1000, len(dot))
for x in range(2):
#    if x == 0:
#        amua_array = ipsi_amua
    if x == 0 :
        amua_array = V_amua
    if x == 1:
        amua_array = contra_amua
    cc = channel[x]
    amua_filter = np.zeros((3, 3, len(t)))
    amua_z = np.zeros((3, 3))
    VE_PT = contra_VE[cc, 0, 1, 0]*100
    VE_ENV = contra_VE[cc, 0, 1, 1]*100
    for ii in range(3):
        for jj in range(3):
            amua_plot = amua_array[cc, ff, dd, ii, jj, :, :]
            amua_plot = filtfilt(bLow,aLow, amua_plot, axis=0, padlen=100)
            amua_plot = np.mean(amua_plot, -1)
            
            # prepare data for AMUA trace 
            amua_trace = amua_plot[int(t_start*Fs_down):int(t_end*Fs_down)]
            amua_trace = amua_trace-np.mean(amua_trace[:6])
            amua_filter[ii, jj, :] = amua_trace         
            
            # prepare data for AMUA heatmap
            amua_heatmap = np.mean(amua_plot[int(t_start*Fs_down):int(t_end_heatmap*Fs_down)])
            amua_z[ii, jj] = amua_heatmap
    amua_max = np.max(amua_filter)
    amua_z = stats.zscore(amua_z)
    
    # plot the AMUA curve use contra example
    if x == 0:
        gs = gs3    
    if x == 1:
        gs = gs1
        
    # plot data grouped by PT_ITD 
    line_style = ['-', '--', ':']
#    color_style = ['#16068A', '#CC4975', '#FDB32E']
    color_style = ['tab:blue', 'tab:orange', 'tab:green']
    PTtext = ['PT_ITD: -0.1', 'PT_ITD: 0', 'PT_ITD: 0.1']
    ENVtext = ['ENV_ITD: -0.1', 'ENV_ITD: 0', 'ENV_ITD: 0.1']
    alpha= [0.65, 0.75, 1]
    for ii in range(3):
        custom_cycler = (cycler('color', ['tab:blue', 'tab:orange', 'tab:green']) *cycler(linestyle = [line_style[ii]]))
#        custom_cycler = (cycler(linestyle = [line_style[ii]]))
        amua_trace = amua_filter[ii, :, :].T
        amua_trace = amua_trace/amua_max
        
        ax = fig1.add_subplot(gs[0, ii])
        ax.set_prop_cycle(custom_cycler)
        ax.plot(t, amua_trace, linewidth=2, alpha = alpha[ii])
        ax.set_ylim(-0.2, 1.1)
        ax.set_xticks([])
        ax.text(0.1, -0.135, PTtext[ii])
        if not ii == 0:
            ax.set_yticks([])
        if x == 0 and ii == 0:
            ax.set_title('ND-Unit#1, VE_PT: '+str("%.1f" % VE_PT)+'%, VE_ENV: '+str("%.1f" % VE_ENV)+'%', loc = 'left')
        if x == 1 and ii == 0:
            ax.set_title('ND-Unit#2, VE_PT: '+str("%.1f" % VE_PT)+'%, VE_ENV: '+str("%.1f" % VE_ENV)+'%', loc = 'left')
        
        l, b, w, h = ax.get_position().bounds
        ax_zoomin = fig1.add_axes([l+0.17, b+0.10, 0.08, 0.08])
        ax_zoomin.set_prop_cycle(custom_cycler)
        ax_zoomin.plot(t, amua_trace, linewidth=2, alpha = alpha[ii])
        ax_zoomin.set_ylim(0.5, 1.1)
        ax_zoomin.set_xlim(17, 31)
        ax_zoomin.set_xticks([])
        ax_zoomin.set_yticks([])
        ax_zoomin.spines['top'].set_visible(False)
        ax_zoomin.spines['right'].set_visible(False)
        
    # plot data grouped by ENV_ITD   
    for jj in range(3):
        custom_cycler = (cycler('linestyle', ['-', '--', ':']) *cycler('color', [color_style[jj]]))
        ax = fig1.add_subplot(gs[1, jj])
        ax.set_prop_cycle(custom_cycler)
        ax.set_ylim(-0.2, 1.1)
        ax.text(0.1, -0.135, ENVtext[jj])
        if not jj == 0:
            ax.set_yticks([])
        if not x ==1:
            ax.set_xticks([])
            
        l, b, w, h = ax.get_position().bounds
        ax_zoomin = fig1.add_axes([l+0.17, b+0.10, 0.08, 0.08])
        ax_zoomin.set_prop_cycle(custom_cycler)
        ax_zoomin.set_ylim(0.5, 1.1)
        ax_zoomin.set_xlim(17, 31)
        ax_zoomin.set_xticks([])
        ax_zoomin.set_yticks([])
        ax_zoomin.spines['top'].set_visible(False)
        ax_zoomin.spines['right'].set_visible(False) 

        for ii in range(3):
            amua_trace = amua_filter[ii, jj, :].T
            amua_trace = amua_trace/amua_max        
            ax.plot(t, amua_trace, linewidth=2, alpha = alpha[ii])
            ax_zoomin.plot(t, amua_trace, linewidth=2, alpha = alpha[ii])
            
    for ii in range(3):
        ax = fig1.add_subplot(gs[0, ii])
        ax.plot(dot_x, dot, '--', color = 'grey')
        
        ax = fig1.add_subplot(gs[1, ii])
        ax.plot(dot_x, dot, '--', color = 'grey')   
#%%        
    # plot heatmap
    heatmap_xticklabel = [-0.1, 0, 0.1]
    heatmap_yticklabel = [0.1, 0, -0.1]
    fig2 = plt.figure(figsize=(6, 5),constrained_layout=True)
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.15, right=0.90, bottom = 0.15, top = 0.85, wspace=0, hspace=0)
    ax_heatmap = fig2.add_subplot(gs[0, 0])
    ax_heatmap.set_title('ND-Unit#1, VE_PT: 30.5%, VE_ENT: 1.3%', loc = 'left', fontsize = 20)
#    ax_heatmap
#    ax = fig1.add_subplot(gs2[x, 0])
#    if x == 0:
#        ax.text(0, -0.2, 'B', fontsize = 14)
    dataframe = pd.DataFrame({'PT_ITD':np.repeat(['-0.1', '0', '0.1'], 3), 'ENV_ITD': np.tile(['-0.1', '0', '0.1'], 3), 'AMUA': amua_z.reshape(3*3)})
    amua_heatmap = dataframe.pivot('ENV_ITD', 'PT_ITD', 'AMUA')
    sns.heatmap(amua_heatmap, ax=ax_heatmap, vmin=np.min(amua_z), vmax=np.max(amua_z), cmap='plasma', annot=True, fmt=".2f", cbar_kws={'label': 'AMUA z score'})
    sns.set(font_scale=1.8)
#    sns.heatmap(amua_heatmap, ax=ax_heatmap, vmin=np.min(amua_z), vmax=np.max(amua_z), cmap='plasma', fmt=".2f")
#    if x == 1:
#        ax.set_xlabel('PT_ITD', loc = 'left')
#        ax.set_ylabel('ENV_ITD', loc = 'bottom')
    if x == 0: 
        ax.set_xlabel('')
        ax.set_ylabel('')
#    ax.set_xticks([])
#    ax.set_yticks([]) 
#    ax.set_xlabel('PT_ITD')
#    ax.set_ylabel('ENV_ITD')
#        

        
# creat legend overall

legend_elements = [Line2D([0], [0], color = 'tab:blue', linestyle = '-', label = 'PT_ITD:-0.1\nENV_ITD:-0.1'),
                   Line2D([0], [0], color = 'tab:blue', linestyle = '--', label = 'PT_ITD:0\nENV_ITD:-0.1'),
                   Line2D([0], [0], color = 'tab:blue', linestyle = ':', label = 'PT_ITD:0.1\nENV_ITD:-0.1'),
                   Line2D([0], [0], color = 'tab:orange', linestyle = '-', label = 'PT_ITD:-0.1\nENV_ITD:0'),
                   Line2D([0], [0], color = 'tab:orange', linestyle = '--', label = 'PT_ITD:0\nENV_ITD:0'),
                   Line2D([0], [0], color = 'tab:orange', linestyle = ':', label = 'PT_ITD:0.1\nENV_ITD:0'),
                   Line2D([0], [0], color = 'tab:green', linestyle = '-', label = 'PT_ITD:-0.1\nENV_ITD:0.1'),
                   Line2D([0], [0], color = 'tab:green', linestyle = '--', label = 'PT_ITD:-0\nENV_ITD:0.1'),
                   Line2D([0], [0], color = 'tab:green', linestyle = ':', label = 'PT_ITD:0.1\nENV_ITD:0.1')]
ax0.legend(handles=legend_elements, loc='center right')

    

    
    
    
    
    
    
    
    
    
