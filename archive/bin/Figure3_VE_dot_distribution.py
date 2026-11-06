#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:07:27 2023

@author: shiyi
"""

import numpy as np
from matplotlib import pyplot as plt
import os
import seaborn as sns
import pandas as pd
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

selection_path = '/disks/CIdata/1_ENVvsPT/selection/ND/_AMUA_Selection/'
VE_path = '/disks/CIdata/1_ENVvsPT/ANOVA/ND/'
file_names = os.listdir(selection_path)
selection_names = [file_name for file_name in file_names if all ([x in file_name for x in ["VE_Selection.npy"]])]
print(selection_names)

selection_all = np.zeros((len(selection_names), 32))
VE_array = np.zeros((len(selection_names), 32, 2, 3, 2))
PT_95 = np.zeros((len(selection_names), 32, 2, 3))
ENV_95 = np.zeros((len(selection_names), 32, 2, 3))
for x in range(len(selection_names)):
    # load data
    selection_name = selection_names[x]
    print(selection_name)
    selection_array = np.load(selection_path+selection_name)
    selection_all[x, :] = selection_array    
    VE_name = selection_name[:-17]+'_AMUA_ExplainedVariance_V2.npy'
    VE = np.load(VE_path+VE_name)
    VE_array[x, :] = VE
    VE_shuffle_name = selection_name[:-17]+'_AMUA_ExplainedVariance_shuffle_V2.npy'
    VE_shuffle = np.load(VE_path+VE_shuffle_name)  
    # check the significance, Ve larger than 95% of the shuffle is significant
    PT_95[x, :, :] = np.percentile(VE_shuffle[:, :, :, 0, :], 95, axis = -1)
    ENV_95[x, :, :] = np.percentile(VE_shuffle[:, :, :, 1, :], 95, axis = -1)
    PT_significance = np.greater(VE_array[:, :, :, :, 0], PT_95)
    ENV_significance = np.greater(VE_array[:, :, :, :, 1], ENV_95)
    
size = len(selection_names)*32
ND_size = size
ND_PTsig = PT_significance
ND_VEarray = VE_array
ND_selection = selection_all
#%%
selection_path = '/disks/CIdata/1_ENVvsPT/selection/NH/_AMUA_Selection/'
VE_path = '/disks/CIdata/1_ENVvsPT/ANOVA/NH/'
file_names = os.listdir(selection_path)
selection_names = [file_name for file_name in file_names if all ([x in file_name for x in ["VE_Selection.npy"]])]
print(selection_names)

selection_all = np.zeros((len(selection_names), 32))
VE_array = np.zeros((len(selection_names), 32, 2, 3, 2))
PT_95 = np.zeros((len(selection_names), 32, 2, 3))
ENV_95 = np.zeros((len(selection_names), 32, 2, 3))
for x in range(len(selection_names)):
    # load data
    selection_name = selection_names[x]
    print(selection_name)
    selection_array = np.load(selection_path+selection_name)
    selection_all[x, :] = selection_array    
    VE_name = selection_name[:-17]+'_AMUA_ExplainedVariance_V2.npy'
    VE = np.load(VE_path+VE_name)
    VE_array[x, :] = VE
    VE_shuffle_name = selection_name[:-17]+'_AMUA_ExplainedVariance_shuffle_V2.npy'
    VE_shuffle = np.load(VE_path+VE_shuffle_name)  
    # check the significance, Ve larger than 95% of the shuffle is significant
    PT_95[x, :, :] = np.percentile(VE_shuffle[:, :, :, 0, :], 95, axis = -1)
    ENV_95[x, :, :] = np.percentile(VE_shuffle[:, :, :, 1, :], 95, axis = -1)
    PT_significance = np.greater(VE_array[:, :, :, :, 0], PT_95)
    ENV_significance = np.greater(VE_array[:, :, :, :, 1], ENV_95)
    
size = len(selection_names)*32
NH_size = size
NH_PTsig = PT_significance
NH_VEarray = VE_array
NH_selection = selection_all

#%% 
'''
Draw only dot plot
ff = 0
dd = 1
'''

fig1 = plt.figure(figsize=(5, 5))
gs = fig1.add_gridspec(1, 1)
ax = fig1.add_subplot(gs[0, 0])
ax.set_title('50ms')

ff = 0
dd = 1
VE_pt = np.reshape(VE_array[: ,:, ff, dd, 0], size)
VE_env = np.reshape(VE_array[: ,:, ff, dd, 1], size)
selection_idx = np.where(np.reshape(selection_all, size) == 1)[0]
VE_pt_selected = VE_pt[selection_idx]
VE_env_selected = VE_env[selection_idx]
dataframe = pd.DataFrame({'VE_pt': VE_pt_selected*100, 'VE_env': VE_env_selected*100})
#sns.scatterplot(data=dataframe, x = 'VE_env', y = 'VE_pt', ax = ax)
#sns.jointplot(data=dataframe, x="VE_env", y="VE_pt", ax = ax)
ax.set_xlim (-3, 80)
ax.set_ylim (-3, 80)
ax.plot(VE_env_selected*100, VE_pt_selected*100, 'o', color = 'tab:blue', markerfacecolor='none')
ax.set_xlabel('Variance explained by ENV_ITD(%)', fontsize = 14)
ax.set_ylabel('Variance explained by PT_ITD(%)', fontsize = 14)
ax.set_facecolor('#E6E6E6')
ax.grid(color='w', linestyle='solid', linewidth = 1.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
#ax.set_xticks([])
#%%
'''
Draw only dot plot
ff = 0
dd = 0
'''
fig1 = plt.figure(figsize=(5, 5))
gs = fig1.add_gridspec(1, 1)
ax = fig1.add_subplot(gs[0, 0])
ax.set_title('10ms')

ff = 0
dd = 0
VE_pt = np.reshape(VE_array[: ,:, ff, dd, 0], size)
VE_env = np.reshape(VE_array[: ,:, ff, dd, 1], size)
selection_idx = np.where(np.reshape(selection_all, size) == 1)[0]
VE_pt_selected = VE_pt[selection_idx]
VE_env_selected = VE_env[selection_idx]
dataframe = pd.DataFrame({'VE_pt': VE_pt_selected*100, 'VE_env': VE_env_selected*100})
#sns.scatterplot(data=dataframe, x = 'VE_env', y = 'VE_pt', ax = ax)
#sns.jointplot(data=dataframe, x="VE_env", y="VE_pt", ax = ax)
ax.set_xlim (-3, 80)
ax.set_ylim (-3, 80)
ax.plot(VE_env_selected*100, VE_pt_selected*100, 'x', color = 'tab:orange', markerfacecolor='none', linewidth = 2)
ax.set_xlabel('Variance explained by ENV_ITD(%)', fontsize = 14)
ax.set_ylabel('Variance explained by PT_ITD(%)', fontsize = 14)
ax.set_facecolor('#E6E6E6')
ax.grid(color='w', linestyle='solid', linewidth = 1.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
#%% 
'''the dot plot of {0.01, 0.05, 0.2} is asigned to three subplot  
   Draw a plot of two variables with bivariate and univariate graphs
'''
widths = [1, 1, 1, 1]
heights = [1, 3]
fig = plt.figure(figsize=(10, 10))
gs = fig.add_gridspec(2, 4,  width_ratios=widths, height_ratios=heights,
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05) 
ax_dot_1 = fig.add_subplot(gs[1, 0])
ax_dot_5 = fig.add_subplot(gs[1, 1])
ax_dot_5.set_yticks([])
ax_dot_2 = fig.add_subplot(gs[1, 2])
ax_dot_2.set_yticks([])
ax_dis_env = fig.add_subplot(gs[0, 0:3])
ax_dis_pt = fig.add_subplot(gs[1, 3])

for dd in range(3):
    VE_pt = np.reshape(VE_array[: ,:, ff, dd, 0], size)    
    VE_env = np.reshape(VE_array[: ,:, ff, dd, 1], size)
    selection_idx = np.where(np.reshape(selection_all, size) == 1)[0]
    VE_pt_selected = VE_pt[selection_idx]
    VE_env_selected = VE_env[selection_idx]
    
    if dd == 0:
        ax_dot_1.plot(VE_env_selected*100, VE_pt_selected*100, 'o', color = 'lightblue', markerfacecolor='none')
        ax_dot_1.set_xlim(-1, 20)
        ax_dot_1.set_ylim(-1, 100)
        ax_dis_env.hist(VE_env_selected*100, 5,  color = "lightblue")
        ax_dis_pt.hist(VE_pt_selected*100, 50,  color = "lightblue", orientation='horizontal')
        
    if dd == 1:
        ax_dot_5.plot(VE_env_selected*100, VE_pt_selected*100, 'o', color = 'orange', markerfacecolor='none')
        ax_dot_5.set_xlim(-1, 20)
        ax_dot_5.set_ylim(-1, 100)    
        ax_dis_env.hist(VE_env_selected*100, 5,  color = "orange")
        ax_dis_pt.hist(VE_pt_selected*100, 50,  color = "orange", orientation='horizontal')
        
    if dd == 2:
        ax_dot_2.plot(VE_env_selected*100, VE_pt_selected*100, 'o', color = 'g', markerfacecolor='none')
        ax_dot_2.set_xlim(-1, 20)
        ax_dot_2.set_ylim(-1, 100)     
        ax_dis_env.hist(VE_env_selected*100, 5,  color = "g")
        ax_dis_pt.hist(VE_pt_selected*100, 50,  color = "g", orientation='horizontal')
#%% 
'''
violin plot
'''

stiRate = [900, 4500]
stiDur = [0.01, 0.05, 0.2]
dataframe = pd.DataFrame(columns=['VE', 'Factor', 'Duration'])
#ff = 0
#dd = 0
#for ff in range(2):
ff = 0
for dd in range(3):
    VE_pt = np.reshape(VE_array[: ,:, ff, dd, 0], size)
    VE_env = np.reshape(VE_array[: ,:, ff, dd, 1], size)
    selection_idx = np.where(np.reshape(selection_all, size) == 1)[0]
    VE_pt_selected = VE_pt[selection_idx]
    VE_env_selected = VE_env[selection_idx]
    for ii in range(2):
        if ii == 0:
            temp = pd.DataFrame({'VE': VE_pt_selected*100, 'Factor': ['PT']*len(VE_pt_selected), 'Duration': np.repeat(stiDur[dd], len(VE_pt_selected))})
            dataframe = pd.concat([dataframe, temp], ignore_index=True)
        else:
            temp = pd.DataFrame({'VE': VE_env_selected*100, 'Factor': ['ENV']*len(VE_pt_selected), 'Duration': np.repeat(stiDur[dd], len(VE_pt_selected))})
            dataframe = pd.concat([dataframe, temp], ignore_index=True)
#        temp = pd.DataFrame({'VE_pt': VE_pt_selected*100, 'VE_env': VE_env_selected*100, 'PulseRate': np.repeat(stiRate[ff], len(VE_pt_selected)), 'Duration': np.repeat(stiDur[dd], len(VE_pt_selected))})
#        dataframe = pd.concat([dataframe, temp], ignore_index=True)
plt.figure()
sns.violinplot(data=dataframe, x = 'Duration', y = 'VE', hue = 'Factor')

#df = sns.load_dataset("titanic")
#sns.violinplot(x=df["age"])
#%% 
'''
violin plot
'''

stiRate = [900, 4500]
stiDur = [0.01, 0.05, 0.2]
Factor = ['ND', 'NH']
dataframe = pd.DataFrame(columns=['Variance Explained Ratio(%)', 'Cohort', 'Duration'])
#ff = 0
#dd = 0
#for ff in range(2):
ff = 0
for d in range(2):
    dd = d+1
    print(stiDur[dd])
    for F in Factor:
        print(F)
        if F == 'ND':
            VE_array = ND_VEarray
            selection_all = ND_selection
        else:
            VE_array = NH_VEarray
            selection_all = NH_selection   
        size = VE_array.shape[0]*32
        VE_pt = np.reshape(VE_array[: ,:, ff, dd, 0], size)
#        VE_env = np.reshape(VE_array[: ,:, ff, dd, 1], size)
        selection_idx = np.where(np.reshape(selection_all, size) == 1)[0]
        VE_pt_selected = VE_pt[selection_idx]
#        VE_env_selected = VE_env[selection_idx]
        temp = pd.DataFrame({'Variance Explained Ratio(%)': VE_pt_selected*100, 'Cohort': F, 'Duration': np.repeat(stiDur[dd], len(VE_pt_selected))})
#        temp = pd.DataFrame({'Variance Explained Ratio(%)': VE_env_selected*100, 'Cohort': F, 'Duration': np.repeat(stiDur[dd], len(VE_env_selected))})
        print(temp)
        dataframe = pd.concat([dataframe, temp], ignore_index=True)
#    for ii in range(2):
#        if ii == 0:
#            temp = pd.DataFrame({'VE': VE_pt_selected*100, 'Factor': ['PT']*len(VE_pt_selected), 'Duration': np.repeat(stiDur[dd], len(VE_pt_selected))})
#            dataframe = pd.concat([dataframe, temp], ignore_index=True)
#        else:
#            temp = pd.DataFrame({'VE': VE_env_selected*100, 'Factor': ['ENV']*len(VE_pt_selected), 'Duration': np.repeat(stiDur[dd], len(VE_pt_selected))})
#            dataframe = pd.concat([dataframe, temp], ignore_index=True)
#        temp = pd.DataFrame({'VE_pt': VE_pt_selected*100, 'VE_env': VE_env_selected*100, 'PulseRate': np.repeat(stiRate[ff], len(VE_pt_selected)), 'Duration': np.repeat(stiDur[dd], len(VE_pt_selected))})
#        dataframe = pd.concat([dataframe, temp], ignore_index=True)
plt.figure()
sns.violinplot(data=dataframe, x = 'Duration', y = 'Variance Explained Ratio(%)', hue = 'Cohort')

#df = sns.load_dataset("titanic")
#sns.violinplot(x=df["age"])
#%%
'''
scatter plot of VE
and distribution
'''
fig1 = plt.figure(figsize=(12, 4),constrained_layout=True)     
gs_scatter = gridspec.GridSpec(1, 3)
gs_scatter.update(left=0.05, right=0.98, bottom = 0.12, top = 0.92, wspace=0.15, hspace=0)
#gs_legend = gridspec.GridSpec(1, 1)
#gs_legend.update(left=0.1, right=1, bottom = 0.1, top = 0.95, wspace=0, hspace=0)
#ax_legend = fig1.add_subplot(gs_legend[0, 0])

## legend
#x = [0]
#y = [0]
#ax_legend.scatter(x, y, s = 80, color = 'tab:blue', facecolors='none', label = 'PT_ITD Sensitive')
#ax_legend.scatter(x, y, s = 80, color = 'tab:red', facecolors='none', label = 'ENV_ITD Sensitive')
#ax_legend.scatter(x, y, s = 80, color = 'purple', facecolors='none', label = 'Both Sensitive')
#ax_legend.scatter(x, y, s = 80, color = 'grey', facecolors='none', label = 'None Sensitive')
#ax_legend.set_xticks([])
#ax_legend.set_yticks([])
#ax_legend.spines['top'].set_visible(False)
#ax_legend.spines['right'].set_visible(False)
#ax_legend.spines['left'].set_visible(False)
#ax_legend.spines['bottom'].set_visible(False)
#ax_legend.legend(loc = 'center right', fontsize = 14)
#ax_legend.set_facecolor("white")


ff = 0
sensNum_bar = np.zeros((2, 3))
for dd in range(3):
#    print(dd)
    PT_temp = np.reshape(VE_array[:, :, ff, dd, 0], size)
    ENV_temp = np.reshape(VE_array[:, :, ff, dd, 1], size)
    selection_idx = np.where(np.reshape(selection_all, size) == 1)[0]
    VE_pt_selected = PT_temp[selection_idx]    
    VE_env_selected = ENV_temp[selection_idx]
    pt_max = np.max(VE_pt_selected)*100
    env_max = np.max(VE_env_selected)*100
    
    PT_sig = np.reshape(PT_significance[:, :, ff, dd], size)
    ENV_sig = np.reshape(ENV_significance[:, :, ff, dd], size)
    PT_sig_selected = PT_sig[selection_idx]
    sensNum_bar[0, dd] = np.where(PT_sig_selected == True)[0].shape[0]
    ENV_sig_selected= ENV_sig[selection_idx]
    sensNum_bar[1, dd] = np.where(ENV_sig_selected == True)[0].shape[0]
    
    ax = fig1.add_subplot(gs_scatter[0, dd]) 
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    if dd == 1:
        ax.set_xlabel('Variance explained by ENV_ITD (%)', fontsize = 14)
        ax.set_title('0.05s')
    if dd == 0:
        ax.set_ylabel('Variance explained by PT_ITD (%)', fontsize = 14)
        ax.set_title('0.01s', fontsize = 14)
    if dd == 2:
        ax.set_title('0.2s', fontsize = 14)
        # legend
        legend_elements = [Line2D([0], [0], marker='o',  color='w', markeredgecolor = 'tab:blue', markerfacecolor='none', label = 'PT_ITD Sensitive', markersize=14),
                           Line2D([0], [0], marker='o',  color='w', markeredgecolor = 'tab:red', markerfacecolor='none', label = 'ENV_ITD Sensitive', markersize=14),
                           Line2D([0], [0], marker='o',  color='w', markeredgecolor = 'purple', markerfacecolor='none', label = 'Both Sensitive', markersize=14),
                           Line2D([0], [0], marker='o',  color='w', markeredgecolor = 'grey', markerfacecolor='none', label = 'None Sensitive', markersize=14)]
        ax.legend(handles=legend_elements, loc='lower right')
        
#    if dd != 0:
#        ax.set_yticks([])
    
#    l, b, w, h = ax.get_position().bounds
#    ax_zoomin = fig1.add_axes([l+0.11, b+0.17, 0.12, 0.16])
#    ax_zoomin.set_xticks([])
#    ax_zoomin.set_yticks([])
#    ax_zoomin.set_xlim(-2, env_max+2) 
#    ax_zoomin.set_ylim(-5, pt_max+5)
             
    for ii in range(len(PT_sig_selected)):
        if PT_sig_selected[ii] == True and ENV_sig_selected[ii] == False:
            ax.scatter(VE_env_selected[ii]*100, VE_pt_selected[ii]*100, facecolors='none', color = 'tab:blue')
#            ax_zoomin.plot(VE_env_selected[ii]*100, VE_pt_selected[ii]*100, 'x', color = 'tab:orange', markerfacecolor='none', linewidth = 2)
    for ii in range(len(PT_sig_selected)):
        if PT_sig_selected[ii] == False and ENV_sig_selected[ii] == False:
#            print(ii)
#            print('PT_VE'+str(VE_pt_selected[ii]))
            ax.scatter(VE_env_selected[ii]*100, VE_pt_selected[ii]*100, facecolors='none', color = 'grey')
#            ax_zoomin.plot(VE_env_selected[ii]*100, VE_pt_selected[ii]*100, 'x', color = 'tab:blue', markerfacecolor='none', linewidth = 2)
    for ii in range(len(PT_sig_selected)):
        if PT_sig_selected[ii] == False and ENV_sig_selected[ii] == True:
            ax.scatter(VE_env_selected[ii]*100, VE_pt_selected[ii]*100, facecolors='none', color = 'tab:red') 
#            print(ii)
#            ax_zoomin.plot(VE_env_selected[ii]*100, VE_pt_selected[ii]*100, 'x', color = 'tab:purple', markerfacecolor='none', linewidth = 5)
    for ii in range(len(PT_sig_selected)):
        if PT_sig_selected[ii] == True and ENV_sig_selected[ii] == True:
            ax.scatter(VE_env_selected[ii]*100, VE_pt_selected[ii]*100, facecolors='none', color = 'purple') 
#            ax_zoomin.plot(VE_env_selected[ii]*100, VE_pt_selected[ii]*100, 'x', color = 'tab:red', markerfacecolor='none', linewidth = 5) 
#%%
'''
use sensNum_bar to plot bar graph
'''
fig1 = plt.figure(figsize=(9, 6),constrained_layout=True)     
gs = gridspec.GridSpec(1, 1)
gs.update(left=0.15, right=0.99, bottom = 0.1, top = 0.9, wspace=0, hspace=0)
ax = fig1.add_subplot(gs[0, 0]) 
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["0.01", "0.05", "0.2"])
y1 = sensNum_bar[0, :]*100/398
y2 = sensNum_bar[1, :]*100/398
x = np.arange(3)
width = 0.4
bar_container1 = ax.bar(x-0.2, y1, width)
bar_container2 = ax.bar(x+0.2, y2, width)
ax.set_ylim(0, 100)
ax.grid()
ax.legend(['PT_ITD', 'ENV_ITD'], fontsize = 18)
ax.set_ylabel('Proportion of ITD sensitive unit (%)', fontsize = 18)
ax.tick_params(axis='both', which='major', labelsize=12)
#ax.set_xlim(-0.5, 2.5)
#%%
'''
Bar plot
'''
#percentage = [333/398, 4/398, 463/507, 6/507]
#dataframe = pd.DataFrame({'cohort': ['ND', 'ND', 'NH', 'NH'], 'Sensitivity': ['PT', 'ENV', 'PT', 'ENV'], '%': percentage})
#sns.barplot(data=dataframe, x="cohort", y="%", hue="Sensitivity")
fig1 = plt.figure(figsize=(9, 6),constrained_layout=True)     
gs = gridspec.GridSpec(1, 1)
gs.update(left=0.15, right=0.99, bottom = 0.1, top = 0.9, wspace=0, hspace=0)
ax = fig1.add_subplot(gs[0, 0]) 
ax.set_xticks([0, 1])
ax.set_xticklabels(["ND", "NH"])
x = np.arange(2)
y1 = [333*100/398, 463*100/507]
y2 = [4*100/398, 6*100/507]
width = 0.4
bar_container1 = ax.bar(x-0.2, y1, width)
bar_container2 = ax.bar(x+0.2, y2, width)
ax.set_ylim(0, 100)
ax.set_xlim(-0.5, 1.75)
ax.legend(['PT_ITD', 'ENV_ITD'], fontsize = 20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.text(-0.27, 85, '84%', fontsize = 20)
ax.text(0.145, 2, '1%', fontsize = 20)
ax.text(0.73, 92, '91%', fontsize = 20)
ax.text(1.15, 2, '1%', fontsize = 20)
ax.set_ylabel('Proportion of ITD sensitive unit (%)', fontsize = 20)
#%%
'''
Bar plot
'''
#percentage = [333/398, 4/398, 463/507, 6/507]
#dataframe = pd.DataFrame({'cohort': ['ND', 'ND', 'NH', 'NH'], 'Sensitivity': ['PT', 'ENV', 'PT', 'ENV'], '%': percentage})
#sns.barplot(data=dataframe, x="cohort", y="%", hue="Sensitivity")
fig1 = plt.figure(figsize=(9, 6),constrained_layout=True)     
gs = gridspec.GridSpec(1, 1)
gs.update(left=0.15, right=0.99, bottom = 0.1, top = 0.9, wspace=0, hspace=0)
ax = fig1.add_subplot(gs[0, 0]) 
ax.set_xticks([0, 1])
ax.set_xticklabels(["0.05s", "0.2s"])
x = np.arange(2)
y1 = [463*100/507, 495*100/507]
y2 = [6*100/507, 26*100/507]
width = 0.4
bar_container1 = ax.bar(x-0.2, y1, width)
bar_container2 = ax.bar(x+0.2, y2, width)
ax.set_ylim(0, 100)
#ax.set_xlim(-0.5, 1.75)
ax.legend(['PT_ITD', 'ENV_ITD'], fontsize = 15)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.text(-0.27, 92, '91%', fontsize = 20)
ax.text(0.145, 2, '1%', fontsize = 20)
ax.text(0.73, 99, '97%', fontsize = 20)
ax.text(1.15, 6, '5%', fontsize = 20)
ax.set_ylabel('Proportion of ITD sensitive unit (%)', fontsize = 20)
#%%
stimPath = '/disks/CIdata/1_ENVvsPT/StimuliData/'
StimulusData = np.load(stimPath+'Stim_ENVvsFS_template.npy')
Fs = 24414.0625
EndIdx = int(Fs*0.051)
Stimulus = StimulusData[:, 1, 0, 2, :EndIdx+1, :]
t = np.arange(0, 0.051, 1/Fs)*1000

'''
scatter plot of VE
and distribution
'''
fig1 = plt.figure(figsize=(12, 6),constrained_layout=True)     
gs_scatter = gridspec.GridSpec(1, 2)
gs_scatter.update(left=0.1, right=0.98, bottom = 0.12, top = 0.92, wspace=0.12, hspace=0)
gs_xlabel = gridspec.GridSpec(1, 1)
gs_xlabel.update(left=0.05, right=0.98, bottom = 0.05, top = 0.92, wspace=0.15, hspace=0)
#gs_xlabel.update(left=0.1, right=1, bottom = 0.1, top = 0.95, wspace=0, hspace=0)
ax_xlabel = fig1.add_subplot(gs_xlabel[0, 0])
ax_xlabel.set_xlabel('Variance explained by ENV_ITD (%)', fontsize = 20)
#gs_legend = gridspec.GridSpec(1, 1)
#gs_legend.update(left=0.1, right=1, bottom = 0.1, top = 0.95, wspace=0, hspace=0)
#ax_legend = fig1.add_subplot(gs_legend[0, 0])

## legend
#x = [0]
#y = [0]
#ax_legend.scatter(x, y, s = 80, color = 'tab:blue', facecolors='none', label = 'PT_ITD Sensitive')
#ax_legend.scatter(x, y, s = 80, color = 'tab:red', facecolors='none', label = 'ENV_ITD Sensitive')
#ax_legend.scatter(x, y, s = 80, color = 'purple', facecolors='none', label = 'Both Sensitive')
#ax_legend.scatter(x, y, s = 80, color = 'grey', facecolors='none', label = 'None Sensitive')
ax_xlabel.set_xticks([])
ax_xlabel.set_yticks([])
ax_xlabel.spines['top'].set_visible(False)
ax_xlabel.spines['right'].set_visible(False)
ax_xlabel.spines['left'].set_visible(False)
ax_xlabel.spines['bottom'].set_visible(False)
#ax_legend.legend(loc = 'center right', fontsize = 14)
#ax_legend.set_facecolor("white")


ff = 0
dur = [0.051, 0.201]
for dd in range(1, 3, 1):
#    print(dd)
    PT_temp = np.reshape(VE_array[:, :, ff, dd, 0], size)
    ENV_temp = np.reshape(VE_array[:, :, ff, dd, 1], size)
    selection_idx = np.where(np.reshape(selection_all, size) == 1)[0]
    VE_pt_selected = PT_temp[selection_idx]    
    VE_env_selected = ENV_temp[selection_idx]
    pt_max = np.max(VE_pt_selected)*100
    env_max = np.max(VE_env_selected)*100
    
    PT_sig = np.reshape(PT_significance[:, :, ff, dd], size)
    ENV_sig = np.reshape(ENV_significance[:, :, ff, dd], size)
    PT_sig_selected = PT_sig[selection_idx]
    ENV_sig_selected= ENV_sig[selection_idx]
    
    ax = fig1.add_subplot(gs_scatter[0, dd-1]) 
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    if dd == 1:
#        ax.set_xlabel('Variance explained by ENV_ITD (%)', fontsize = 14)
        ax.set_title('0.05s', fontsize = 20)
    if dd == 1:
        ax.set_ylabel('Variance explained by PT_ITD (%)', fontsize = 20)
#        ax.set_title('0.01s', fontsize = 14)
    if dd == 2:
        ax.set_title('0.2s', fontsize = 20)
        # legend
        legend_elements = [Line2D([0], [0], marker='o',  color='w', markeredgecolor = 'tab:blue', markerfacecolor='none', label = 'PT_ITD Sensitive', markersize=20),
                           Line2D([0], [0], marker='o',  color='w', markeredgecolor = 'tab:red', markerfacecolor='none', label = 'ENV_ITD Sensitive', markersize=20),
                           Line2D([0], [0], marker='o',  color='w', markeredgecolor = 'purple', markerfacecolor='none', label = 'Both Sensitive', markersize=20),
                           Line2D([0], [0], marker='o',  color='w', markeredgecolor = 'grey', markerfacecolor='none', label = 'None Sensitive', markersize=20)]
        ax.legend(handles=legend_elements, loc='lower right', fontsize = 18)
        
#    if dd != 0:
#        ax.set_yticks([])
    
    l, b, w, h = ax.get_position().bounds
    ax_zoomin = fig1.add_axes([l+0.1, b+0.55, 0.3, 0.2])
    ax_zoomin.set_xticks([])
    ax_zoomin.set_yticks([])
    t = np.arange(0, dur[dd-1], 1/Fs)
    ax_zoomin.plot(t, StimulusData[0, dd, 1, 1, :int(dur[dd-1]*Fs)+1, :])
    
    ax_zoomin.spines['top'].set_visible(False)
    ax_zoomin.spines['right'].set_visible(False)
#    ax_zoomin.spines['left'].set_visible(False)
#    ax_zoomin.spines['bottom'].set_visible(False)
    
    ax_zoomin.set_xlim(0, 0.2)
    
#    ax_zoomin.set_xlim(-2, env_max+2) 
#    ax_zoomin.set_ylim(-5, pt_max+5)
             
    for ii in range(len(PT_sig_selected)):
        if PT_sig_selected[ii] == True and ENV_sig_selected[ii] == False:
            ax.scatter(VE_env_selected[ii]*100, VE_pt_selected[ii]*100, facecolors='none', color = 'tab:blue')
#            ax_zoomin.plot(VE_env_selected[ii]*100, VE_pt_selected[ii]*100, 'x', color = 'tab:orange', markerfacecolor='none', linewidth = 2)
    for ii in range(len(PT_sig_selected)):
        if PT_sig_selected[ii] == False and ENV_sig_selected[ii] == False:
#            print(ii)
#            print('PT_VE'+str(VE_pt_selected[ii]))
            ax.scatter(VE_env_selected[ii]*100, VE_pt_selected[ii]*100, facecolors='none', color = 'grey')
#            ax_zoomin.plot(VE_env_selected[ii]*100, VE_pt_selected[ii]*100, 'x', color = 'tab:blue', markerfacecolor='none', linewidth = 2)
    for ii in range(len(PT_sig_selected)):
        if PT_sig_selected[ii] == False and ENV_sig_selected[ii] == True:
            ax.scatter(VE_env_selected[ii]*100, VE_pt_selected[ii]*100, facecolors='none', color = 'tab:red') 
#            print(ii)
#            ax_zoomin.plot(VE_env_selected[ii]*100, VE_pt_selected[ii]*100, 'x', color = 'tab:purple', markerfacecolor='none', linewidth = 5)
    for ii in range(len(PT_sig_selected)):
        if PT_sig_selected[ii] == True and ENV_sig_selected[ii] == True:
            ax.scatter(VE_env_selected[ii]*100, VE_pt_selected[ii]*100, facecolors='none', color = 'purple') 
#            ax_zoomin.plot(VE_env_selected[ii]*100, VE_pt_selected[ii]*100, 'x', color = 'tab:red', markerfacecolor='none', linewidth = 5) 
#%% plot distribution of PT_sensivie
fig1 = plt.figure(figsize=(9, 4),constrained_layout=True)
gs_dis = gridspec.GridSpec(1, 2)
gs_dis.update(left=0.05, right=0.95, bottom = 0.1, top = 0.9, wspace=0.15, hspace=0)
ax1 = fig1.add_subplot(gs_dis[0, 0]) 
ax2 = fig1.add_subplot(gs_dis[0, 1]) 
            
stiDur = [0.01, 0.05, 0.2]       
ff = 0
dataframe_PT = pd.DataFrame(columns = ['VE', 'Factor', 'Dur'])
dataframe_ENV = pd.DataFrame(columns = ['VE', 'Factor', 'Dur'])


for dd in range(3):
    VE_pt = np.reshape(VE_array[: ,:, ff, dd, 0], size)
    VE_env = np.reshape(VE_array[: ,:, ff, dd, 1], size)
    selection_idx = np.where(np.reshape(selection_all, size) == 1)[0]
    VE_pt_selected = VE_pt[selection_idx]
    VE_env_selected = VE_env[selection_idx]    
    PT_sig = np.reshape(PT_significance[:, :, ff, dd], size)
    ENV_sig = np.reshape(ENV_significance[:, :, ff, dd], size)
    PT_sig_selected = PT_sig[selection_idx]
    ENV_sig_selected= ENV_sig[selection_idx]
    PT_sensitive_idx = np.where(PT_sig_selected)[0]
    ENV_sensitive_idx = np.where(ENV_sig_selected)[0]
    VE_pt_selected = VE_pt_selected[PT_sensitive_idx]
    VE_env_selected = VE_env_selected[ENV_sensitive_idx]   
    for ii in range(2):
        if ii == 0:
            temp = pd.DataFrame({'VE': VE_pt_selected*100, 'Factor': ['PT']*len(VE_pt_selected), 'Dur': np.repeat(stiDur[dd], len(VE_pt_selected))})
            dataframe_PT = pd.concat([dataframe_PT, temp], ignore_index=True)
        if ii == 1:
            temp = pd.DataFrame({'VE': VE_env_selected*100, 'Factor': ['ENV']*len(VE_env_selected), 'Dur': np.repeat(stiDur[dd], len(VE_env_selected))})
            dataframe_ENV = pd.concat([dataframe_ENV, temp], ignore_index=True)
#palette = ['#1f77b4', '#ff7f0e', '#2ca02c']
   
sns.histplot(dataframe_PT, ax = ax1, x="VE", hue="Dur", element="step", palette=['#1f77b4', '#ff7f0e', '#2ca02c'])
sns.histplot(dataframe_ENV, ax = ax2, x="VE", hue="Dur", element="step", palette=['#1f77b4', '#ff7f0e', '#2ca02c'])

#%% plot distribution of PT_sensitive on ND and NH cohort
ff = 0
dd = 1
fig1 = plt.figure(figsize=(8, 6),constrained_layout=True)
gs = gridspec.GridSpec(1, 1)
gs.update(left=0.1, right=0.9, bottom = 0.1, top = 0.9, wspace=0, hspace=0)
ax1 = fig1.add_subplot(gs[0, 0]) 
ax1.tick_params(labelsize=20)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
for ii in range(2):
    if ii == 0: # ND
#        plt.figure()
        VE_pt = np.reshape(ND_VEarray[: ,:, ff, dd, 0], ND_size)
        selection_idx = np.where(np.reshape(ND_selection, ND_size) == 1)[0]
        VE_pt_selected = VE_pt[selection_idx]
        ND_total = len(VE_pt_selected)
        
        n_ND, x_ND, _ = ax1.hist(VE_pt_selected*100, alpha = 0.2, facecolor="tab:blue", edgecolor='None', label = 'ND')
        n_ND, x_ND, _ = ax1.hist(VE_pt_selected*100, facecolor="None", edgecolor='tab:blue', lw = 1.5)
    if ii == 1: # NH
#        plt.figure()
        VE_pt = np.reshape(NH_VEarray[: ,:, ff, dd, 0], NH_size)
        selection_idx = np.where(np.reshape(NH_selection, NH_size) == 1)[0]
        VE_pt_selected = VE_pt[selection_idx]
        NH_total = len(VE_pt_selected)
        
        n_NH, x_NH, _ = ax1.hist(VE_pt_selected*100, alpha = 0.2, facecolor="tab:orange", edgecolor='None', label = 'NH')
        n_NH, x_NH, _ = ax1.hist(VE_pt_selected*100, facecolor="None", edgecolor='tab:orange', lw = 1.5)
bin_centers_ND = 0.5*(x_ND[1:]+x_ND[:-1])
bin_centers_NH = 0.5*(x_NH[1:]+x_NH[:-1])  
ax1.plot(bin_centers_ND, n_ND/ND_total, '-o', label = 'ND')   
ax1.plot(bin_centers_NH, n_NH/NH_total, '-o', label = 'NH')  
ax1.legend(fontsize = 20)
#%%
bin_centers_ND = 0.5*(x_ND[1:]+x_ND[:-1])
bin_centers_NH = 0.5*(x_NH[1:]+x_NH[:-1])    
plt.figure()
plt.plot(bin_centers_ND, n_ND/ND_total, '-o', label = 'ND')   
plt.plot(bin_centers_NH, n_NH/NH_total, '-o', label = 'NH')  
plt.legend(fontsize = 20)
plt.ylabel('Normalized Count', fontsize = 20)
plt.xlabel('Variance Explained by PT-ITD', fontsize = 20)
plt.tick_params(labelsize=20)
plt.title('0.05s', fontsize = 20)

