#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 14:03:37 2025

@author: shiyi

Plot out the stimulus shape
"""
import config  # data/figure folders, see config.py
eaSource = config.EPHYS_LIB+'/'
from sys import path
path.append(eaSource)
import ClickTrainLibrary as cl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy import stats
# read in stim.csv
stim = pd.read_csv(config.DATA_ROOT+'/stim.csv')
#%%
'''
load in stimulus waveform
'''
# defaultStimParams={
#         'duration (s)': 0.05,
#         'clickRate (Hz)': 900,
#         'ABL (dB)': 5,
#         'ITD (ms)': 0,
#         'env ITD (ms)': 0,
#         'envelope': 'hanning',
#         'ILD (dB)': 0,
#         'Nloop': 1}
# # defaultStimParams = stim.iloc[0]
# # stimulusClass=cl.ITD_TWFclicktrain
# stimulusClass=cl.ITD_ILD_ENV_clicktrain
# stim=stimulusClass()
# for stimPar in defaultStimParams.keys():
#     stim.stimParams[stimPar]=defaultStimParams[stimPar]
# stim.ready()
# # stim.sampleRate = 24414.0625
# # p = stim.plot()
# stimulus = stim.sounds
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

def search_multiunit(df, duration, rate, condition=None, threshold=0, metric='VEpt'):
    """
    Filter recording sites based on specified criteria
    
    Parameters:
    -----------
    df : pandas.DataFrame
        recordingSites dataframe
    duration : float
        Stimulus duration (0.01, 0.05, 0.2)
    rate : float
        Stimulus rate (900.0, 4500.0)
    condition : str, optional
        Experimental condition (if None, no filtering by condition)
    threshold : float
        VE threshold (default 0, i.e., VE > 0)
    metric : str
        Metric type 'VEpt' or 'VEenv' (default 'VEpt')
    
    Returns:
    --------
    filtered_df : pandas.DataFrame
        Filtered dataframe
    """
    # Construct column name
    col_name = f'Dur{duration}_Rate{rate}.0_{metric}'
    
    # Check if column exists
    if col_name not in df.columns:
        raise ValueError(f"Column {col_name} not found in dataframe")
    
    # Start filtering
    filtered_df = df.copy()
    
    # Filter by condition (if specified)
    if condition is not None:
        filtered_df = filtered_df[filtered_df['condition'] == condition]
    
    # Filter by VE value
    filtered_df = filtered_df[filtered_df[col_name] > threshold]
    
    return filtered_df.reset_index(drop=True)

def wilcoxonTestfor(swps, stimPar, conditionToCheck):
    swpIdx=findSweepsFor(conditionToCheck.copy(), stimPar)
    swpTest = swps[swpIdx]
    sampleRate = swpTest[0].sampleRate
    nchans = swpTest[0].signal.shape[1]
    # duration = conditionToCheck[0][1]
    duration = stimPar.iloc[swpIdx[0]]['duration (s)']
    selectionArray = []
    resultsArray = []
    for cc in range(nchans):
        signal = poolChannel(swpTest, cc)
        responseWinAvg = signal[:int(duration*sampleRate), :].mean(axis = 0)
        baselineWinAvg = signal[-int(duration*sampleRate):, :].mean(axis = 0)
        results = stats.wilcoxon(responseWinAvg, baselineWinAvg)
        resultsArray.append(results[1])
        if results[1] < 0.05:
            selectionArray.append(1)
        else:
            selectionArray.append(0)
    return resultsArray, selectionArray

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
#%%
'''
'''
import RZ2ephys as ep
analysisRoot = config.ANALYSIS_ROOT+'/Salt_Beaker_CSV_Files/'
dataRoot = config.DATA_ROOT
dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_Salt_Beaker_Agar.csv')
recordingSites = pd.read_csv(analysisRoot+'/recordingsiteInventory_Salt_Beaker_Agar.csv')

ITD = [-0.1, 0, 0.1]
dur = 0.05
pps = 900
t_start = 0.002
t_end = 0.1
paramsDict = {'duration (s)': [0.01, 0.05, 0.2], 'clickRate (Hz)': [900.0, 4500.0], 'ITD (ms)': [-0.1, 0.0, 0.1], 'env ITD (ms)': [-0.1, 0.0, 0.1]}
# stimPath = '/ephys/4/CIproject/0_ENVvsPTephys/Analysis/'
# stimdata=np.load(stimPath+'Stim_ENVvsFS.npy')

# result1 = search_multiunit(recordingSites, duration = dur, rate = pps, threshold = 0.1, metric = 'VEpt')
#%%
'''
give filename, read in data
'''
filename = 'Beaker1_1_ITD_ENV_vs_PT_CI_1'
anim = 'SaltBeaker_ENVvsPT_AGAR'
chanNum = 29
responsive = recordingSites[(recordingSites['fileName'] == filename)&(recordingSites['Channel'] == chanNum)]['Responsive'].iloc[0]
print(responsive)
swps, stimPar = ep.readEphysFile(dataRoot+'/'+anim+'/'+filename)
nup = ep.nonUniqueParams(stimPar)
swpsClean, stimPar = ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-5]+'_Clean')
swpsAMUA, stimPar = ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-5]+'_AMUA')
#%% 
plt.figure('fig1')
plt.figure('fig2')
sampleRateAMUA = swpsAMUA[1].sampleRate
sampleRateclean = swpsClean[1].sampleRate
chanNum = 23
for ii in range(3):
    conditions = [('duration (s)', 0.01), ('clickRate (Hz)',900), ('ITD (ms)', ITD[ii]), ('env ITD (ms)', 0)]
    idxs_prev = findSweepsFor(conditions.copy(), stimPar)
    
    #plot amua
    tAMUA = np.arange(0, 0.01+0.05, 1/sampleRateAMUA)
    
    signalAMUA = poolChannel(swpsAMUA[idxs_prev], chanNum)
    # signal = signal-signal[0]
    signalAMUA = signalAMUA.mean(axis = 1)
    signalAMUA = signalAMUA-signalAMUA[0]
    plt.figure('fig1')
    plt.plot(tAMUA, signalAMUA[:len(tAMUA)])
    
    #plot clean sig
    tclean = np.arange(0, 0.01+0.05, 1/sampleRateclean)
    
    signalclean = poolChannel(swpsClean[idxs_prev], chanNum)
    signalclean = signalclean.mean(axis = 1)
    signalclean = signalclean-signalclean[0]
    plt.figure('fig2')
    plt.plot(tclean, signalclean[:len(tclean)])
    
#%%
stimParCLEAN = pd.read_csv(dataRoot+'/'+anim+'/'+filename[:-5]+'_Clean.csv')
# nup = ep.nonUniqueParams(stimPar)
p = ep.permutations(nup, reverse = True)
# remove_keys = LeftOverArtefact (nup, p, chanNum, stimParCLEAN)
Headername = f'ABS_STD_DIV_chan{chanNum}'  # Ensure exact match
abs_std_columns = [col for col in stimParCLEAN.columns if col == Headername]

Headername_baseline = f'DIV_baseline{chanNum}'  # Ensure exact match
abs_std_columns_baseline = [col for col in stimParCLEAN.columns if col == Headername_baseline]

# Create a new DataFrame with only ABS_STD columns
df_abs_std = stimParCLEAN[abs_std_columns]
df_abs_std_baseline = stimParCLEAN[abs_std_columns_baseline]

# Calculate the 95th percentile of the baseline
percentile_value = np.percentile(df_abs_std_baseline, 95)
leftoverart = np.where(df_abs_std > 2 * percentile_value)[0]
cleanart = np.where(df_abs_std <= 2 * percentile_value)[0]
#%%
duration = [0.01, 0.05, 0.2]
pulserate = [900, 4500]
sampleRate = swpsClean[1].sampleRate

for ff in range(2):
    for dd in range(3):
        fig, axs = prepare3X3Plot()
        t = np.arange(0, duration[dd]+0.05, 1/sampleRate)
        for ii in range(9):
            xx = ii % 3
            yy = ii//3
            pt_itd = ITD[yy]
            env_itd = ITD[xx]
            ax = axs[ii]
            
            conditions = [('duration (s)', duration[dd]), ('clickRate (Hz)',pulserate[ff]), ('ITD (ms)', pt_itd), ('env ITD (ms)', env_itd)]
            idxs_prev = findSweepsFor(conditions.copy(), stimPar)-1
            idxs = np.array([item for item in idxs_prev if item not in leftoverart])
            if len(idxs) == 0:
                continue
            print(idxs)
            idxs = idxs+1
            signal = poolChannel(swpsClean[idxs], chanNum)
            # signal_plot = signal-signal[-200:, :].mean(axis = 0)
            signal_plot = signal-signal[0, :]
            signal_plot = signal_plot[:len(t), :]
            
            ax.plot(t, signal_plot[:len(t), :])
#%%
if len(leftoverart) == 0:
    remove_keys = np.empty(0, dtype=object)
    # return remove_keys

elif len(leftoverart) > 0:
    Artefact_check = []  # Holds combinations with too much artefact
    for ind, stim in enumerate(p):
        idxs_prev = ep.sweepIndecesFor(stimParCLEAN, list(nup.keys()), stim)
    # check if there is residual artefact:
        idxs = np.array([item for item in idxs_prev if item not in leftoverart])
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
#%%
conditions=[('duration (s)',dur), ('clickRate (Hz)',pps), ('ITD (ms)',0), ('env ITD (ms)', 0)]
idxs_prev =findSweepsFor(conditions.copy(), stimPar)
signal_test = poolChannel(swpsClean[idxs_prev], chanNum)
sampleRate = swpsClean[1].sampleRate
t = np.arange(t_start, t_end, 1/sampleRate)*1000
signal_plot = signal_test[int(t_start*sampleRate):int(t_end*sampleRate), :]
signal_baseline = signal_plot[-100:, :].mean(axis = 0)
signal_plot = signal_plot-signal_baseline
plt.figure()
plt.plot(t, signal_plot[:, 5])

#%%
'''
ITD tunning curve
'''
result1 = search_multiunit(recordingSites, duration = dur, rate = pps, threshold = 0.1, metric = 'VEpt')
result_response = result1[result1['Responsive'] == 1]
for i, info in result1.iterrows():
    filename = info['fileName']
    chanNum = info['Channel']
    anim = dataFiles[(dataFiles['fileName'] == filename)]['animDir'].iloc[0]
    
    AMUAdf = pd.read_csv(dataRoot+'/'+anim+'/'+filename[:-6]+'_AMUA.csv')
    cols_to_extract = ['duration (s)', 'clickRate (Hz)', 'ABL (dB)', 'ITD (ms)', 
                   'env ITD (ms)', 'envelope', 'ILD (dB)', 'Nloop', 'timeStamp',
                   f'AMUA_Avg_Chan{chanNum}']
    newAMUAdf = AMUAdf[cols_to_extract]
    
    plt.figure()
    col_name = f'Dur{dur}_Rate{pps}.0_VEpt'
    vept = round(info[col_name]*100, 2)
    plt.suptitle(f'{filename[:-6]}, ch{chanNum}, VEpt: {vept}')
    itd_tunning = []
    for itd in ITD:
        df = newAMUAdf[(newAMUAdf['duration (s)'] == dur) & (newAMUAdf['clickRate (Hz)'] == pps) & (newAMUAdf['ITD (ms)'] == itd)]
        amuaavg = df[f'AMUA_Avg_Chan{chanNum}'].mean()
        itd_tunning.append(amuaavg)
    plt.plot(ITD, itd_tunning)
    plt.xticks(ticks=ITD, labels=['-0.1', '0', '0.1'])
    plt.xlabel('PT ITD')
    plt.ylabel('AMUAavg')

    plt.savefig(f'{config.FIGURE_ROOT}/review/{filename}_ch{chanNum}.png')
    plt.close('all')
#%%
'''
give filename and channel number
plot out AMUA trace under PT ITD = 0
'''
filename = '2022_01_11_ENVvsPT_5_P02_LIC_1.ephys'
chanNum = 0
t_start = 0.002
t_end = 0.1

anim = dataFiles[(dataFiles['fileName'] == filename)]['animDir'].iloc[0]

swpsAMUA, stimPar = ep.readEphysFile(dataRoot+'/'+anim+'/'+filename[:-6]+'_AMUA')

# simply check the AMUA trace under PT ITD = 0
plt.figure()
for itd in ITD:
    conditions=[('duration (s)',dur), ('clickRate (Hz)',pps), ('ITD (ms)',itd)]
    idxs_prev =findSweepsFor(conditions.copy(), stimPar)
    signal = poolChannel(swpsAMUA[idxs_prev], chanNum)
    sampleRate = swpsAMUA[0].sampleRate        
    # t = np.arange(0, duration+0.05, 1/sampleRate)
    t = np.arange(t_start, t_end, 1/sampleRate)*1000
    Wn = 2*200/sampleRate
    bLow,aLow = butter(2, Wn, 'lowpass')
    amua_plot = filtfilt(bLow,aLow, signal, axis=0, padlen=100).mean(axis = 1)
    baseline = np.mean(amua_plot[-100:])
    amua_plot = amua_plot-baseline
    # amua_plot = amua_plot/(max_y_values[i]*1)
    plt.plot(t, amua_plot[int(t_start*sampleRate):int(t_end*sampleRate)], label = f'ptitd {itd}')
    plt.legend()    
    plt.suptitle(f'{filename[:-6]}, ch{chanNum}')















