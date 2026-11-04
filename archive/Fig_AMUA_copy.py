#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:45:40 2025

@author: shiyi

AMUA figure for ENVvsPT ephys paper
The figure has 6 subplots for 3 cohort

"""
import pandas as pd
from sys import path
path.append('/home/colliculus/behaviourBoxes/software/ratCageProgramsV3/')
import RZ2ephys as ep
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
dataRoot = '/ephys/2/0_ENVvsPT/Data'
analysisRoot = '/ephys/2/0_ENVvsPT/Analysis'
figureRoot = '/ephys/2/0_ENVvsPT/Figure'
#%%
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
#%%
# load datadileinventory
dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_V3.csv')
recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventoryV3.csv')

duration = 0.05
clickRate = 900.0
# read in example AMUA
# cohorts = ['NH', 'ND', 'ITD+']
# cohorts = ['NH', 'NH', 'NH']
cohorts = ['NH']
itd = [+0.1, 0, -0.1]
t_start = 0.002
t_end = 0.1
#%%
# # animNums = [4, 6, 2]
# # penNums = [0, 3, 2]
# # chanNums = [13, 23, 17]
# animNums = [4, 4, 0]
# penNums = [0, 1, 3]
# chanNums = [13, 5, 25]
filename = '20220822_ENVvsPT_5_LIC_P2_1.ephys'
chanNum = 18
#%%
all_y_values = []
VEpt = []
VEenv = []
# max_y_values_temp = []
max_y_values = []
for i, cohort in enumerate(cohorts):
    max_y_values_temp = []
    # if i == 0:
    if True:
        # # animNum = int(input(f"Enter animal number for {cohort} cohort: "))
        # # penNum = int(input(f"Enter penetration number for {cohort} cohort: "))
        # # chanNum = int(input(f"Enter channel number for {cohort} cohort: "))
        # # animNum = 0
        # # penNum = 0
        # # chanNum = 0
        # animNum = animNums[i]
        # penNum = penNums[i]
        # chanNum = chanNums[i]
        
        anim = dataFiles[(dataFiles['fileName'] == filename)]['animDir'].iloc[0]
        
        # anim = dataFiles[(dataFiles['condition'] == cohort)]['animDir'].unique()[animNum]
        # filename = dataFiles[(dataFiles['animDir'] == anim) & (dataFiles['Cleaned'] == 1)].iloc[penNum]['fileName']
        df = recordingSites[recordingSites['fileName'] == filename]
        df = df[df['Channel'] == chanNum]
        keyName = f'Dur{duration}_Rate{clickRate}_'
        VEpt.append(round(df[f'{keyName}VEpt'].values[0]*100, 2))
        VEenv.append(round(df[f'{keyName}VEenv'].values[0]*100, 2))
        swpsAMUA, stimPar = ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-6]+'_AMUA')
        sampleRate = swpsAMUA[0].sampleRate        
        t = np.arange(t_start, t_end, 1/sampleRate)*1000
        Wn = 2*300/sampleRate
        bLow,aLow = butter(2, Wn, 'lowpass')
        
        pt_itd = 0
        for env_itd in itd:
            conditions=[('duration (s)',duration), ('clickRate (Hz)',clickRate), ('ITD (ms)',pt_itd), ('env ITD (ms)',env_itd)]
            idxs_prev =findSweepsFor(conditions.copy(), stimPar)
            signal = poolChannel(swpsAMUA[idxs_prev], chanNum)
            amua_plot = filtfilt(bLow,aLow, signal, axis=0, padlen=100).mean(axis = 1)
            baseline = np.mean(amua_plot[-100:])
            amua_plot = amua_plot-baseline
            all_y_values.append(amua_plot[int(t_start*sampleRate):int(t_end*sampleRate)])
            max_y_values_temp.append(np.max(amua_plot[int(t_start*sampleRate):int(t_end*sampleRate)]))

        env_itd = 0
        for pt_itd in itd:
            conditions=[('duration (s)',duration), ('clickRate (Hz)',clickRate), ('ITD (ms)',pt_itd), ('env ITD (ms)',env_itd)]
            idxs_prev =findSweepsFor(conditions.copy(), stimPar)
            signal = poolChannel(swpsAMUA[idxs_prev], chanNum)
            amua_plot = filtfilt(bLow,aLow, signal, axis=0, padlen=100).mean(axis = 1)
            baseline = np.mean(amua_plot[-100:])
            amua_plot = amua_plot-baseline
            all_y_values.append(amua_plot[int(t_start*sampleRate):int(t_end*sampleRate)])
            max_y_values_temp.append(np.max(amua_plot[int(t_start*sampleRate):int(t_end*sampleRate)]))
            
    max_y_values.append(np.max(max_y_values_temp))
        
        


all_y_values = np.concatenate(all_y_values)
y_min = np.min(all_y_values)
y_max = np.max(all_y_values)

#%%
# Define colors with alpha
AD_Acute_Color = [0.2, 0.1, 0.4, 0.2]
ND_Acute_Color = [0.2, 0.5, 0.6, 0.2]
ND_Chronic_Color = [0.1, 0.0, 1.0, 0.2]

bg_colors = [AD_Acute_Color, ND_Acute_Color, ND_Chronic_Color]

#%%
# fig = plt.figure(figsize=(8, 12), constrained_layout=False)  
# # Add a common y label to the left middle of the figure
# fig.text(0.03, 0.5, 'AMUA (μV)', 
#          rotation=90, 
#          va='center', 
#          ha='center',
#          fontsize=12)
# gs = fig.add_gridspec(6, 2)
# gs.update(left=0.15, right=0.95,  
#           bottom=0.08, top=1,   
#           hspace=0)              
# axs = []

# # First create all axes and set their basic properties
# for ii in range(6):
#     ax = fig.add_subplot(gs[ii, 0])
#     axs.append(ax)
#     ax.set_ylim(y_min*0.8, y_max*1.1)
    
#     # Set background color
#     cohort_index = ii // 2
#     ax.set_facecolor(bg_colors[cohort_index])
    
#     ax.tick_params(axis='y', rotation=45)
    
#     # ax.text(0.95, 0.95, 'ITDpt = 0',
#     #     transform=ax.transAxes,  
#     #     ha='right',              
#     #     va='top',                
#     #     fontsize=10)
    
#     if ii % 2 == 0:
#         ax.set_xticklabels([])
#         ax.text(0.95, 0.95, 'PT ITD = 0',
#             transform=ax.transAxes,  
#             ha='right',              
#             va='top',                
#             fontsize=12)
#         # Add titles for even-numbered subplots
#         plot_index = ii // 2
#         if plot_index == 0:
#             # ax.set_title(f'AD + acute CI, EVE_pt: {VEpt[0]}%, EVE_env: {VEenv[0]}%')
#             ax.set_title(f'AD + acute CI, EVE_env: {VEenv[0]}%')
#         elif plot_index == 1:
#             # ax.set_title(f'ND + acute CI, EVE_pt: {VEpt[1]}%, EVE_env: {VEenv[1]}%')
#             ax.set_title(f'AD + acute CI, EVE_env: {VEenv[1]}%')
#         elif plot_index == 2:
#             # ax.set_title(f'ND + chronic CI, EVE_pt: {VEpt[2]}%, EVE_env: {VEenv[2]}%')
#             ax.set_title(f'AD + acute CI, EVE_env: {VEenv[1]}%')
            
#     if ii %2 == 1:
#         ax.text(0.95, 0.95, 'ENV ITD = 0',
#             transform=ax.transAxes,  
#             ha='right',              
#             va='top',                
#             fontsize=12)
#         plot_index = ii // 2
#         if plot_index == 0:
#             ax.set_title(f'EVE_pt: {VEpt[0]}%')
#         if plot_index == 1:
#             ax.set_title(f'EVE_pt: {VEpt[1]}%')
#         if plot_index == 2:
#             ax.set_title(f'EVE_pt: {VEpt[2]}%')            
        

# # Define spacing parameters
# height = 0.125  # Adjusted height
# group_gap = 0.07  # Adjusted gap between groups
# no_gap = 0.025  # no gap between paired plots

# # Calculate positions from top to bottom
# top_position = 0.84  # Lowered top position to keep within bounds

# for ii in range(6):
#     ax = axs[ii]
    
    
    
#     # Calculate y position
#     if ii == 0:
#         y_pos = top_position
#     elif ii == 1:
#         y_pos = top_position - height - no_gap
#     elif ii == 2:
#         y_pos = top_position - 2*height - group_gap
#     elif ii == 3:
#         y_pos = top_position - 3*height - group_gap - no_gap
#     elif ii == 4:
#         y_pos = top_position - 4*height - 2*group_gap
#     else:  # ii == 5
#         y_pos = top_position - 5*height - 2*group_gap - no_gap
    
#     # Set position
#     ax.set_position([
#         0.1,  # x position
#         y_pos,  # y position
#         0.65,  # width
#         height  # height
#     ])
    
#     if ii in [0, 2, 4]:
#         ax.tick_params(axis='x', length=0)
#     if ii == 5:
#         ax.set_xlabel('Time(ms)', fontsize = 12)
        
#%% try 2*3 layout
fig = plt.figure(figsize=(6.69, 3.8), constrained_layout=False)  
# Add a common y label to the left middle of the figure
# fig.text(0.02, 0.5, 'AMUA (μV)', 
#          rotation=90, 
#          va='center', 
#          ha='center',
#          fontsize=12)
fig.text(0.02, 0.5, 'Normalized AMUA', 
         rotation=90, 
         va='center', 
         ha='center',
         fontsize=10)
fig.text(0.5, 0.03, 'Time(ms)', 
         rotation=0, 
         va='center', 
         ha='center',
         fontsize=10)
gs = fig.add_gridspec(2, 3)
gs.update(left=0.08, right=0.98,  
          bottom=0.12, top=0.9,   
          hspace=0.23, wspace = 0.1)              
axs = []

# First create all axes and set their basic properties
for ii in range(6):
    xx = ii%2
    yy = int(ii/2)
    ax = fig.add_subplot(gs[xx, yy])
    axs.append(ax)
    # ax.set_ylim(y_min*0.8, y_max*1.1)
    ax.set_ylim(-0.1, 1)
    
    # Set background color
    # cohort_index = ii // 2
    # ax.set_facecolor(bg_colors[cohort_index])
    
    ax.tick_params(axis='y', rotation=45, labelsize=9)
    ax.tick_params(axis='x', labelsize=9)
    
    if yy != 0:
        ax.tick_params(labelbottom=True, labelleft=False, labelsize=9)
    # ax.text(0.95, 0.95, 'ITDpt = 0',
    #     transform=ax.transAxes,  
    #     ha='right',              
    #     va='top',                
    #     fontsize=10)
    
    if xx == 0:
        ax.set_xticklabels([])
        ax.text(0.05, 0.95, 'PT ITD = 0',
            transform=ax.transAxes,  
            ha='left',              
            va='top',                
            fontsize=10)
        # Add titles for even-numbered subplots
        plot_index = ii // 2
        if plot_index == 0:
            # ax.set_title(f'AD + acute CI, EVE_pt: {VEpt[0]}%, EVE_env: {VEenv[0]}%')
            # ax.set_title(f'AD + acute CI, VE ENV: {VEenv[0]}%')
            ax.set_title(f'Multi Unit#1\n$\\it{{VE(ENV)}}$: {VEenv[0]}%', fontsize = 10)
        elif plot_index == 1:
            # ax.set_title(f'ND + acute CI, EVE_pt: {VEpt[1]}%, EVE_env: {VEenv[1]}%')
            # ax.set_title(f'ND + acute CI, VE ENV: {VEenv[1]}%')
            ax.set_title(f'Multi Unit#2\n$\\it{{VE(ENV)}}$: {VEenv[1]}%', fontsize = 10)
        elif plot_index == 2:
            # ax.set_title(f'ND + chronic CI, EVE_pt: {VEpt[2]}%, EVE_env: {VEenv[2]}%')
            # ax.set_title(f'ND + chronic CI, VE ENV: {VEenv[2]}%')
            ax.set_title(f'Multi Unit#3\n$\\it{{VE(ENV)}}$: {VEenv[2]}%', fontsize = 10)
            
    if xx == 1:
        ax.text(0.05, 0.95, 'ENV ITD = 0',
            transform=ax.transAxes,  
            ha='left',              
            va='top',                
            fontsize=9)
        plot_index = ii // 2
        if plot_index == 0:
            ax.set_title(f'$\\it{{VE(PT)}}$: {VEpt[0]}%', fontsize = 10)
        if plot_index == 1:
            ax.set_title(f'$\\it{{VE(PT)}}$: {VEpt[1]}%', fontsize = 10)
        if plot_index == 2:
            ax.set_title(f'$\\it{{VE(PT)}}$: {VEpt[2]}%', fontsize = 10)            
         
#%%
for i, cohort in enumerate(cohorts):
    # if i == 2:
    if True:
        xx = i%2
        yy = int(i/2)
        # # animNum = int(input(f"Enter animal number for {cohort} cohort: "))
        # # penNum = int(input(f"Enter penetration number for {cohort} cohort: "))
        # # chanNum = int(input(f"Enter channel number for {cohort} cohort: "))
        # # animNum = 0
        # # penNum = 0
        # # chanNum = 0
        # animNum = animNums[i]
        # penNum = penNums[i]
        # chanNum = chanNums[i]
        
        anim = dataFiles[(dataFiles['fileName'] == filename)]['animDir'].iloc[0]
        
        # anim = dataFiles[(dataFiles['condition'] == cohort)]['animDir'].unique()[animNum]
        # filename = dataFiles[(dataFiles['animDir'] == anim) & (dataFiles['Cleaned'] == 1)].iloc[penNum]['fileName']
        swpsAMUA, stimPar = ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-6]+'_AMUA')
        df = recordingSites[recordingSites['fileName'] == filename]
        df = df[df['Channel'] == chanNum]
        keyName = f'Dur{duration}_Rate{clickRate}_'
        VEpt = round(df[f'{keyName}VEpt'].values[0]*100, 1)
        VEenv = round(df[f'{keyName}VEenv'].values[0]*100, 1)
        # get ready for AMUA plot
        sampleRate = swpsAMUA[0].sampleRate        
        # t = np.arange(0, duration+0.05, 1/sampleRate)
        t = np.arange(t_start, t_end, 1/sampleRate)*1000
        Wn = 2*200/sampleRate
        bLow,aLow = butter(2, Wn, 'lowpass')
        
        # # set zoom in
        # axins1 = inset_axes(axs[i*2], width="30%", height="30%", 
        #             loc='upper right', borderpad=1)
        # axins2 = inset_axes(axs[i*2+1], width="30%", height="30%", 
        #             loc='upper right', borderpad=1)
        
        # when ptitd = 0, loop all envitd
        pt_itd = 0
        for env_itd in itd:
            conditions=[('duration (s)',duration), ('clickRate (Hz)',clickRate), ('ITD (ms)',pt_itd), ('env ITD (ms)',env_itd)]
            if env_itd == 0.1:
                print(conditions)
            idxs_prev =findSweepsFor(conditions.copy(), stimPar)
            signal = poolChannel(swpsAMUA[idxs_prev], chanNum)
            amua_plot = filtfilt(bLow,aLow, signal, axis=0, padlen=100).mean(axis = 1)
            baseline = np.mean(amua_plot[-100:])
            amua_plot = amua_plot-baseline
            amua_plot = amua_plot/(max_y_values[i]*1)
            # amua_plot = signal[:, 11]
            # for x in range(20):
                
            #     plt.figure()
            #     plt.plot(signal[:, x])
            # plt.title(label=f'env ITD {env_itd}')
            if env_itd == 0.1:
                label = f'ENV ITD +{int(env_itd*0.8*1000)} µs'
            elif env_itd == -0.1:
                label = f'ENV ITD {int(env_itd*0.8*1000)} µs'
            elif env_itd == 0:
                label = f'ENV ITD {env_itd} µs'
            axs[i*2].plot(t, amua_plot[int(t_start*sampleRate):int(t_end*sampleRate)], label=label)
            # axs[i*2].plot(t, amua_plot[int(t_start*sampleRate):int(t_end*sampleRate)]/(max_y_values[i]*1.2), label=f'ENV ITD {env_itd}')
            axs[i*2].axvline(x=duration*1000, color='k', linestyle='--',alpha = 0.3)            
            # axins1.plot(amua_plot[int(0.015*sampleRate):int(0.035*sampleRate)])
            axs[i*2].set_yticks([0, 0.5, 1])
            
        # when envitd = 0, loop all ptitd
        env_itd = 0
        for pt_itd in itd:
            conditions=[('duration (s)',duration), ('clickRate (Hz)',clickRate), ('ITD (ms)',pt_itd), ('env ITD (ms)',env_itd)]
            idxs_prev =findSweepsFor(conditions.copy(), stimPar)
            signal = poolChannel(swpsAMUA[idxs_prev], chanNum)
            amua_plot = filtfilt(bLow,aLow, signal, axis=0, padlen=100).mean(axis = 1)
            baseline = np.mean(amua_plot[-100:])
            amua_plot = amua_plot-baseline
            amua_plot = amua_plot/(max_y_values[i]*1)
            # amua_plot = signal[:, 3]
            if pt_itd == 0.1:
                label = f'PT ITD +{int(pt_itd*0.8*1000)} µs'
            elif pt_itd == -0.1:
                label = f'PT ITD {int(pt_itd*0.8*1000)} µs'
            elif pt_itd == 0:
                label = f'PT ITD {pt_itd} µs'
            axs[i*2+1].plot(t, amua_plot[int(t_start*sampleRate):int(t_end*sampleRate)], label=label)
            # axs[i*2+1].plot(t, amua_plot[int(t_start*sampleRate):int(t_end*sampleRate)]/(max_y_values[i]*1.2), label=f'PT ITD {pt_itd}')
            axs[i*2+1].axvline(x=duration*1000, color='k', linestyle='--', alpha = 0.3) 
            # axins2.plot(amua_plot[int(0.015*sampleRate):int(0.035*sampleRate)])
            axs[i*2+1].set_yticks([0, 0.5, 1])
        
        # Remove inset labels if desired
        # axins1.set_xticklabels([])
        # axins1.set_yticklabels([])
        # axins2.set_xticklabels([])
        # axins2.set_yticklabels([])
        # axins1.set_facecolor('none')
        # axins2.set_facecolor('none')
        # axins1.spines['top'].set_visible(False)
        # axins1.spines['right'].set_visible(False)
        # axins2.spines['top'].set_visible(False)
        # axins2.spines['right'].set_visible(False)
        # axins1.tick_params(axis='y', length=0)
        # axins2.tick_params(axis='y', length=0)
        # axins1.tick_params(axis='x', length=0)
        # axins2.tick_params(axis='x', length=0)
        
        if i == 2:
            axs[i*2].legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.02, 0.9), framealpha=0)
            axs[i*2+1].legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.02, 0.9), framealpha=0)
            

fig.savefig('/ephys/2/0_ENVvsPT/Figure/'+'Figure3.png', dpi = 600)
            
        
            
    
    
    
    
    
    
