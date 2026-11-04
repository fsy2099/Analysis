#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:18:15 2025

@author: shiyi
"""
import config  # data/figure folders, see config.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import plasma
analysisRoot = config.ANALYSIS_ROOT

dataFiles = pd.read_csv(analysisRoot+'/datafileInventory.csv')
recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventory.csv')
#%%
cohorts = ['NH', 'ND', 'ITD+']
clickRate = [900.0, 4500.0]
duration = [0.01, 0.05, 0.2]
AD_Acute_Color = [0.2, 0.1, 0.4]
ND_Acute_Color = [0.2, 0.5, 0.6]
ND_Chronic_Color = [0.1, 0.0, 1.0]
bg_colors = [AD_Acute_Color, ND_Acute_Color, ND_Chronic_Color]
alphas = [0.2, 0.4, 0.6]
# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.sans-serif': ['DejaVu Sans', 'Arial']
# })
# plt.rcParams['font.family'] = 'Times New Roman'
#%%
fig = plt.figure(figsize=(6.69, 5.95), constrained_layout=False)
gs = fig.add_gridspec(3, 3)
gs.update(left=0.09, right=0.85, bottom=0.12,
          top=0.92, wspace=0.1, hspace=0.1)
axs = []
for ii in range(9):
    xx = ii % 3
    yy = ii//3
    ax = fig.add_subplot(gs[yy, xx])
    
    # Set x and y limits for all subplots
    ax.set_xlim([-5, 105])
    # ax.set_xlim([-1, 20])
    ax.set_ylim([-5, 105])
    
    # Set background color
    # ax.set_facecolor(bg_colors[yy])
    
    # Hide all ticks first
    ax.tick_params(labelbottom=False, labelleft=False, labelsize=9)
    
    # ax.set_facecolor((*bg_colors[yy], alphas[xx]))
    
    # Show left ticks only for the leftmost plots
    if xx == 0:
        ax.tick_params(labelleft=True, labelsize=9)
    #     if yy == 0:
    #         ax.set_ylabel('AD + acute CI', fontsize = 12)
    #     if yy == 1:
    #         ax.set_ylabel('ND + acute CI', fontsize = 12)
    #     if yy == 2:
    #         ax.set_ylabel('ND + chronic CI', fontsize = 12)
    
    # Show bottom ticks only for the bottom plots
    if yy == 2:
        ax.tick_params(labelbottom=True)
        
    if yy == 0:
        if xx == 0:
            ax.set_title('10ms, 900 pps', fontsize = 10)
        if xx == 1:
            ax.set_title('50ms, 900 pps', fontsize = 10)
        if xx == 2:
            ax.set_title('200ms, 900 pps', fontsize = 10)   
            
    
    axs.append(ax)
    
# Add left label (vertical)
fig.text(0.03, 0.5, 'VE(PT) in %', 
         rotation=90,  # vertical text
         va='center',   # vertical alignment
         ha='center',   # horizontal alignment
         fontsize = 10,
         style = 'italic')   

# Add bottom label (horizontal)
fig.text(0.5, 0.04, 'VE(ENV) in %', 
         va='center',   # vertical alignment
         ha='center',   # horizontal alignment
         fontsize = 10, 
         style = 'italic')   

fig.text(0.76, 0.90, 'AD + acute CI', 
         rotation=0,
         va='center',   # vertical alignment
         ha='center',   # horizontal alignment
         fontsize = 10,
         color = bg_colors[0]) 

fig.text(0.76, 0.625, 'ND + acute CI', 
         rotation=0,
         va='center',   # vertical alignment
         ha='center',   # horizontal alignment
         fontsize = 10,
         color = bg_colors[1]) 

fig.text(0.755, 0.35, 'ND + chronic CI', 
         rotation=0,
         va='center',   # vertical alignment
         ha='center',   # horizontal alignment
         fontsize = 10,
         color = bg_colors[2]) 

#%%
cohorts = ['NH', 'ND', 'ITD+']
clickRate = [900.0, 4500.0]
duration = [0.01, 0.05, 0.2]
markers = ['o', 's', '^', 'v', 'D', 'p', 'h', '8']
# removeList = ['CI2306_ITD_ENV_vs_PT_CI_1', 'CI2220_ITD-_ENV_vs_PT_CI_1', '2023_02_23']
alpha = 1 

for corhNum, cohort in enumerate(cohorts):
    if True:
        datalist = {i: {'env': [], 'pt': []} for i in range(3)}
        anims = dataFiles[dataFiles['condition']== cohort]['animDir'].unique()
        colors = plasma(np.linspace(0, 1, len(anims)))
        np.random.shuffle(colors)
        for animNum, anim in enumerate(anims):
            # if anim not in removeList:
            if True:
                filenames = dataFiles[(dataFiles['animDir'] == anim) & (dataFiles['Cleaned'] == 1)]
                for penNum, filename in enumerate(filenames['fileName']):
                    # if filename != '20220824_ENVvsPT_5_LIC_P4_1.ephys':
                    if True:
                        sub_recordingSites = recordingSites[(recordingSites['fileName'] == filename)]
                        for ccidx, cc in enumerate(clickRate):
                            if ccidx == 0:
                                for ddidx, dd in enumerate(duration):
                                    columnName = f'Dur{dd}_Rate{cc}'
                                    VEheader = [f'{columnName}_VEpt', f'{columnName}_VEenv', 'Channel']
                                    xy = sub_recordingSites[VEheader]
                                    pt = xy[VEheader[0]]
                                    env = xy[VEheader[1]]
                                    
                                    ax = axs[corhNum*3+ddidx]
                                    
                                    # Modified scatter plot parameters
                                    ax.scatter(env*100, pt*100, 
                                              s=10,  # point size
                                              facecolors=colors[animNum],  # fill color
                                              edgecolors=colors[animNum],  # edge color
                                              alpha=alpha-animNum*0.1,  # transparency
                                              label=f'Anim {anim}')
                                    
                                    datalist[ddidx]['env'].extend(env*100)
                                    datalist[ddidx]['pt'].extend(pt*100)
                                    
                                    if ddidx == 2:
                                        handles, labels = ax.get_legend_handles_labels()
                                        seen = set()
                                        unique_handles = []
                                        unique_labels = []
                                        rat_num = 1
                                        for h, l in zip(handles, labels):
                                            if l not in seen:
                                                seen.add(l)
                                                unique_handles.append(h)
                                                unique_labels.append(f'Rat {rat_num}')
                                                rat_num += 1
                                        ax.legend(unique_handles, unique_labels,
                                                 fontsize=10,
                                                 bbox_to_anchor=(1.01, 1.0),
                                                 loc='upper left',
                                                 labelspacing=0.2)


        # # Calculate and add percentile lines for each duration
        # for dd in range(3):
        #     ptdataall = datalist[dd]['pt']
        #     # Remove NaN values
        #     ptdataall_clean = [x for x in ptdataall if not np.isnan(x)]
        #     envdataall = datalist[dd]['env']
        #     envdataall_clean = [x for x in envdataall if not np.isnan(x)]
        #     # Calculate 95th percentiles
        #     env_95 = np.percentile(envdataall_clean, 95)
        #     pt_95 = np.percentile(ptdataall_clean, 95)
            
        #     # Add percentile lines
        #     ax = axs[corhNum*3+dd]
        #     # ax.axvline(x = env_95, color='gray', linestyle='--', alpha=0.5)
        #     # ax.axhline(y = pt_95, color='gray', linestyle='--', alpha=0.5)
            
        #     # label n
        #     n = len(ptdataall_clean)
        #     ax.text(0.95, 0.05, f'n = {n}', 
        #             transform=ax.transAxes,
        #             ha='right', va='bottom',
        #             fontsize=8, color='black')
            
        # Hard-coded n from Table 1 (900 pps only)
        # Rows: AD+acute CI, ND+acute CI, ND+chronic CI
        # Cols: 10ms, 50ms, 200ms
        n_table = [
            [703, 679, 677],   # AD + acute CI
            [681, 654, 706],   # ND + acute CI
            [991, 919, 1009],  # ND + chronic CI
        ]

        for dd in range(3):
            ptdataall = datalist[dd]['pt']
            ptdataall_clean = [x for x in ptdataall if not np.isnan(x)]
            envdataall = datalist[dd]['env']
            envdataall_clean = [x for x in envdataall if not np.isnan(x)]
            env_95 = np.percentile(envdataall_clean, 95)
            pt_95 = np.percentile(ptdataall_clean, 95)
            
            ax = axs[corhNum*3+dd]
            # ax.axvline(x = env_95, color='gray', linestyle='--', alpha=0.5)
            # ax.axhline(y = pt_95, color='gray', linestyle='--', alpha=0.5)
            
            # Annotate n from Table 1
            ax.text(0.95, 0.05, f'n = {n_table[corhNum][dd]}', 
                    transform=ax.transAxes,
                    ha='right', va='bottom',
                    fontsize=10, color='black')
            
fig.savefig(config.FIGURE_ROOT+'/'+'Figure4.png', dpi = 600)