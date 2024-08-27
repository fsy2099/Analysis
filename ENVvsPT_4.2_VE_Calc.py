#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 14:16:55 2022

@author: Shiyi
"""

#%%
import pandas as pd
from sys import platform
import numpy as np
import numpy.matlib
import os
import matplotlib  
from matplotlib import pyplot as plt
import sys
from numpy.lib.stride_tricks import as_strided
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, csd, windows, welch, get_window
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM

if platform == "darwin":
    libPath='/Users/fei/Documents/CI_projects/DataAnalysis'
    sys.path.append(libPath)
elif platform =="linux":
    libPath='/home/colliculus/behaviourBoxes/software/ratCageProgramsV2/'
    sys.path.append(libPath)
# import RZ2ephys as ep
from matplotlib.gridspec import GridSpec

#give path of predict artifact and original data
if platform == "darwin":  
    os.chdir('/Users/fei/Documents/CI_projects/20220127_Analysis')
    fname = '20220127_ENVvsPT_5_P10_1'   
elif platform == "linux":
    Sig_path = '/disks/CIdata/1_ENVvsPT/AMUA/NH/'
    Results_path = '/disks/CIdata/1_ENVvsPT/ANOVA/NH/'
    selection_path = '/disks/CIdata/1_ENVvsPT/selection/NH/_AMUA_Selection/'
    fig_path = '/disks/CIdata/1_ENVvsPT/ANOVA/NH/'
    avg_path = '/disks/CIdata/1_ENVvsPT/selection/NH/_Avg_Window/'
elif platform == "win32":
    Sig_path = 'C:\\Users\\shiyi\\Documents\\1_Project\\ENVvsPT\\data\\'
    results_path = 'C:\\Users\\shiyi\\Documents\\1_Project\\ENVvsPT\\data\\'  
    avg_path = 'C:\\Users\\shiyi\\Documents\\1_Project\\ENVvsPT\\data\\' 

#%% Compute explained variance by hand
Fs = 24414.0625
Fs_downsample = 2000
file_names = os.listdir(Sig_path)
sig_names = [file_name for file_name in file_names if all([x in file_name for x in [ "_OriginSigArray_AMUA.npy"]])]
#clean_names = [file_name for file_name in file_names if all([x in file_name for x in ["_CleanSig.npy"]])]
print(sig_names)
stiDur = [0.01, 0.05, 0.2]
#ff = 0
#dd = 1
#t_start = 0.005
#t_end = stiDur[dd]+t_start
#IdxStart = round(t_start*Fs_downsample)
#Idxend = round(t_end*Fs_downsample)
for xx in range(len(sig_names)):
    sig_name = sig_names[xx]
    print(sig_name)
    amua_array = np.load(Sig_path+sig_name)
    average_window = np.load(avg_path+sig_name[:-4]+'_Avg_Window.npy')
    selection_name = sig_name[:-9]+'_VE_Selection.npy'
    selection_array = np.load(selection_path+selection_name)
    if amua_array.shape[-1] != 15 and amua_array.shape[-1] != 30:
        print('Worng trial length!')
        continue
    
#    amua_avg = np.mean(amua_array[:, ff, dd, :, :, IdxStart:Idxend, :], 3)
    VE_penetration = np.zeros((32, 2, 3, 2))
    VE_shuffle = np.zeros((32, 2, 3, 2, 1000))
    for cc in range(32):
        if selection_array[cc] == 0:
            continue
        print(cc)
        for ff in range(2):
            for dd in range(3):     
#                t_start = 0.005
#                t_end = stiDur[dd]+t_start
#                IdxStart = round(t_start*Fs_downsample)
#                Idxend = round(t_end*Fs_downsample)
                IdxStart = int(average_window[cc, ff, dd, 0])
                Idxend = int(average_window[cc, ff, dd, 1])
                
                amua_temp = np.mean(amua_array[cc, ff, dd, :, :, IdxStart:Idxend, :], 2)
                ntrials = amua_temp.shape[-1]
                '''
                compute the explained variance of original AMUA array
                '''
                # calculate total_avg
                amua_total_avg_temp = np.mean(amua_temp)
                # calculate SSpt
                amua_pt_avg = np.mean(np.mean(amua_temp, 1), -1)
                amua_total_avg = np.repeat(amua_total_avg_temp, 3)
                SSpt = 3*ntrials*sum(pow(amua_pt_avg-amua_total_avg, 2))
                df_pt = 3-1
                MSpt = SSpt/df_pt         
                # calculate SSenv
                amua_env_avg = np.mean(np.mean(amua_temp, 0), -1)
                SSenv = 3*ntrials*sum(pow(amua_env_avg-amua_total_avg, 2))   
                df_env = 3-1
                MSenv = SSenv/df_env
                # calculate SStotal
                amua_total_avg = np.reshape(np.repeat(np.mean(amua_temp), 3*3*ntrials), (3, 3, ntrials))
                SStotal = sum(sum(sum(pow(amua_temp-amua_total_avg, 2))))
                df_total = ntrials*9-1
                MStotal = SStotal/df_total
                # calculate SSwithin
                amua_within_avg = np.repeat(np.mean(amua_temp, -1)[:, :, np.newaxis], ntrials, axis = -1)
                SS_within = sum(sum(sum(pow(amua_temp-amua_within_avg, 2))))
                # calculate explained variance ratio
                VEpt = SSpt/SStotal
        #        VEpt = SSpt/(SStotal-SS_within)
                VEenv = SSenv/SStotal
                VE_penetration[cc, ff, dd, 0] = VEpt
                VE_penetration[cc, ff, dd, 1] = VEenv  
                '''
                use stats.anova_lm to compute the two way anova results
                '''
                dataframe = pd.DataFrame({'PT_ITD': np.repeat(np.repeat(['-0.1', '0', '0.1'], 3), ntrials), 'ENV_ITD': np.tile(np.repeat(['-0.1', '0', '0.1'], ntrials), 3), 'AMUA': amua_temp.reshape(3*3*ntrials)})
                model = ols('AMUA ~ C(PT_ITD)+C(ENV_ITD)+C(PT_ITD):C(ENV_ITD)', data = dataframe).fit()
                result = sm.stats.anova_lm(model, type=1)
                '''
                shuffle the AMUA array and compute the explained variance
                '''
                amua_long = np.reshape(amua_temp, 3*3*ntrials)
                idx = np.arange(3*3*ntrials)
                for tt in range(1000):
        #            print(tt)
                    np.random.shuffle(idx)
                    amua_shuffle = (np.take_along_axis(amua_long, idx, axis=0)).reshape(3, 3, ntrials)
                    # calculate SSpt_shuffle
                    amua_pt_avg_shuffle = np.mean(np.mean(amua_shuffle, 1), -1)
                    amua_total_avg = np.repeat(amua_total_avg_temp, 3)
                    SSpt_shuffle = 3*ntrials*sum(pow(amua_pt_avg_shuffle-amua_total_avg, 2))
                    # calculate SSenv_shuffle
                    amua_env_avg_shuffle = np.mean(np.mean(amua_shuffle, 0), -1)
                    SSenv_shuffle = 3*ntrials*sum(pow(amua_env_avg_shuffle-amua_total_avg, 2))
                    # calculate SStotal
                    amua_total_avg = np.reshape(np.repeat(np.mean(amua_temp), 3*3*ntrials), (3, 3, ntrials))
                    SStotal_shuffle = sum(sum(sum(pow(amua_shuffle-amua_total_avg, 2))))
                    # calculate explained variance ratio
                    VEpt_shuffle = SSpt_shuffle/SStotal_shuffle
                    VEenv_shuffle = SSenv_shuffle/SStotal_shuffle
                    VE_shuffle[cc, ff, dd, 0, tt] = VEpt_shuffle
                    VE_shuffle[cc, ff, dd, 1, tt] = VEenv_shuffle        
    np.save(Results_path+sig_name[:-4]+'_ExplainedVariance.npy', VE_penetration)
    np.save(Results_path+sig_name[:-4]+'_ExplainedVariance_shuffle.npy', VE_shuffle)
