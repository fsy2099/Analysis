#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 15:43:22 2022

artifact rejection
read raw response from Sig_path
stimulus waveform from stimPath
save the clean signal to results_path

@author: shiyi
"""
from sys import platform
import numpy as np
import numpy.matlib
import os 
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import as_strided
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, csd, windows, welch, get_window
# give signal path
if platform == "darwin":  
    Sig_path = '/disks/CIdata/1_ENVvsPT/2_ND/2023_03_03/Results_Data/'
    results_path = '/disks/CIdata/1_ENVvsPT/2_ND/2023_03_03/Results_Data/'   
elif platform == "linux":
    Sig_path = '/disks/CIdata/1_ENVvsPT/2_ND/2023_03_03/Results_Data/'
#    Sig_path = '/media/shiyi/CIdata/2023_02_16/'
    results_path = '/disks/CIdata/1_ENVvsPT/2_ND/2023_03_03/Results_Data/'
#    results_path = '/media/shiyi/CIdata/2023_02_16/'
# load stimulus waveform    
if platform == "darwin":
    stimPath = '/Users/fei/Documents/CI_projects/StimData_v1/'
elif platform == "linux":
    stimPath = '/disks/CIdata/1_ENVvsPT/StimuliData/'
#    stimPath = '/media/shiyi/CIdata/StimuliData/'
StimulusData = np.load(stimPath+'Stim_ENVvsFS_template.npy')
#%%   
def _check_arg(x, xname):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError('%s must be one-dimensional.' % xname)
    return x

def crosscorrelation(x, y, maxlag):
    """
    Cross correlation with a maximum number of lags.
    `x` and `y` must be one-dimensional numpy arrays with the same length.
    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(a)-maxlag-1:len(a)+maxlag]
    The return vaue has length 2*maxlag + 1.
    """
    x = _check_arg(x, 'x')
    y = _check_arg(y, 'y')
    py = np.pad(y.conj(), 2*maxlag, mode='constant')
    T = as_strided(py[2*maxlag:], shape=(2*maxlag+1, len(y) + 2*maxlag),
                    strides=(-py.strides[0], py.strides[0]))
    px = np.pad(x, maxlag, mode='constant')
    return T.dot(px)

def wienerfilt1(X,Y,N,Fs):
    Y1 = Y-np.mean(Y)
    X1 = X-np.mean(X)
    H1 = crosscorrelation(Y1,X1,N)
    H = (H1 / np.var(X))/Fs
    
    H = H - np.mean(H[:N])
    H = H[N:]
    return H

def concatenation(X):
    Y = np.reshape(X.T, [1, X.shape[0]*X.shape[1]])
    Y = Y[0]
    return Y
    
def FindPeak2Peak(X):
    if not int(X[0]) == 0:
        X = X-X[0]
    p2p = np.max(X) - np.min(X)
    return p2p

def CalcSF(X, Y, n):
    idx = np.arange(np.argmax(Y[:n])-12, np.argmax(Y[:n])+13)
    SF = FindPeak2Peak(X[idx])/FindPeak2Peak(Y[idx])
    return SF

# N is the kasier window
def CalcCSD(S1, S2, N, Fs): 
    f, H = csd(S1, S2, Fs, window = N, nperseg = len(N))
    H = 10*np.log10(np.abs(H))
    return H

def CalcARR(S1, S2, C1, C2, N, Fs):
    window = get_window(('kaiser', 5), N)
    ARR = CalcCSD(C1, C2, window, Fs)/CalcCSD(S1, S2, window, Fs)
    return ARR

# O is the kasier window order; F is the fundemental frequence, Fs is the sampling rate
def SelectFre(O, F, Fs):
    step = int(np.around(Fs/(2*int(O/2+1))))
    Fre = np.arange(F, int(Fs/2), F)
    idx = np.around(Fre/step)
    idx = [int(x) for x in idx]
    return idx

def CalcPSD(S, N, Fs):
    f, H = welch(S, Fs, window = N, nperseg = len(N))
    H = 10*np.log10(np.abs(H))
    return H

# S1 is the signal trial signal, S2 is the mean across all trial
#def CalcSNR(S1, S2, N, Fs):
#    window = get_window(('kaiser', 5), N)
#    PSD = CalcPSD(S1, window, Fs)
#    CSD = CalcCSD(S1, S2, window, Fs)
#    SNR = PSD-CSD/CSD
#    return SNR

# A is the artifact length
def CalcFFT(S, A, Fs):
#    insig = np.concatenate((S, np.zeros((2, int(Fs/2) - Artifact_length))), axis = 1)
    fft = np.abs(np.fft.rfft(S))
    return fft
#def CalcFFT(S, A, Fs):
#    padding = np.zeros((int(Fs/2) - Artifact_length))
#    insig = np.concatenate((S, padding))
#    fft = np.abs(np.fft.rfft(S))
#    return fft

def CalcSNR(S, F, Fs):
    fre = np.around(np.linspace(0,Fs/2,len(S)))
    SNR = np.mean(S[np.where((fre<F+10) & (fre>F-10))[0]])/np.mean(S[np.where((fre<F+100) & (fre>F+20))[0]])
    return SNR
#%% 
file_names = os.listdir(Sig_path)
sig_names = [file_name for file_name in file_names if all([x in file_name for x in ["_OriginSigArray.npy"]])]
print(sig_names)
Fs = 24414.0625
WienerFilterOrder = 25
#nsamples = int(Fs*0.4) # Change for GUI recording
#%%
for sig_name in sig_names: 
#sig_name = sig_names[0]
#    dic = np.load(Sig_path+sig_name, allow_pickle = True).item()
    sig_array = np.load(Sig_path+sig_name)
    print(sig_name)
    nsamples = sig_array.shape[-2]
    print(nsamples/Fs)
    if nsamples/Fs < 0.35:
        continue
#    sig_array = dic['OriginSig_array']
#    sig_concatenate = dic['OriginSig_concatenate']
    ntrials = sig_array.shape[-1]
#    nsamples = sig_array.shape[-2]
#    Fs = dic['Fs']
#    stm = dic['Stimulus']
    stiDur = [0.01, 0.05, 0.2]
    stiRate = [900, 4500]
    stiITD = [-0.1, 0, 0.1]
    stienvITD = [-0.1, 0, 0.1]
    
    reject_mark = np.zeros((32, len(stiDur),len(stiITD),len(stienvITD)),dtype = 'float32')
    H0_array = np.zeros((32,len(stiDur),len(stiITD),len(stienvITD),WienerFilterOrder+1),dtype = 'float32')
    clean_array = np.zeros((32,len(stiRate),len(stiDur),len(stiITD),len(stienvITD),nsamples,ntrials),dtype = 'float32')
    SF_array = np.zeros((32, len(stiRate),len(stiDur),len(stiITD),len(stienvITD)), dtype = 'float32')
    Predict_array = np.zeros((32, len(stiRate),len(stiDur),len(stiITD),len(stienvITD), nsamples))
    fft_array = np.zeros((32, len(stiRate),len(stiDur),len(stiITD),len(stienvITD),int(np.around(Fs/4)), ntrials), dtype = 'float32')
    SNR_array = np.zeros((32, len(stiRate),len(stiDur),len(stiITD),len(stienvITD), ntrials), dtype = 'float32')
    CleanArtifact = np.zeros((32, len(stiDur), len(stiITD), len(stienvITD), ntrials), dtype = 'float32')
    KaiserWinOrder = 500
    for cc in range(32):
        print('Chan'+str(cc+1))
        for dd in range(3):
            for ii in range(3):
                for jj in range(3):
                    StimParam = [stiDur[dd], stiITD[ii], stienvITD[jj]]
                    print(StimParam)
                    Artifact_length = int(np.around(Fs*stiDur[dd]))
                    # 900pps                        
                    sig_9 = concatenation(sig_array[cc, 0, dd, ii, jj, :, :])                   
                    stim = StimulusData[0, dd, ii, jj, :nsamples, 0]
                    stim_9 = np.matlib.repmat(stim,1,ntrials).T[:, 0]
                    stim_9[stim_9 < 0] = 0
                    # 1. calculate the initnal wiener filter kernel(H0)
                    H0 = wienerfilt1(stim_9, sig_9, WienerFilterOrder, Fs/WienerFilterOrder)
                    H0 = H0-H0[0]
                    H0_array[cc, dd, ii, jj, :] = H0
                    # 2. use H0 convolve with stimulus signal to get the predicted artifact
                    predict_9 = np.convolve(H0, stim_9)
                    Predict_array[cc, 0, dd, ii, jj, :Artifact_length] = predict_9[:Artifact_length]
                    # 3. there is shift between origin signal and predicted artifact sometimes, use correlate method to aline the signal
                    # use the first artifact of first trial
                    # shift predict artifact to match the original signal
                    correlate_9 = np.correlate(sig_9[: Artifact_length], predict_9[: Artifact_length], 'full')
                    max_idx = np.argmax(correlate_9[Artifact_length-5: Artifact_length+5])
                    shift = max_idx-(5-1)
                    if shift < 0:
                        predict_9 = predict_9[np.abs(shift): len(sig_9)+np.abs(shift)]
                    elif shift > 0:
                        padding = np.zeros((shift))
                        predict_9 = np.concatenate((padding, predict_9[: len(sig_9)-shift]))     
                    # 4. Calculate the scale factor
                    ScaleFactor_9 = CalcSF(np.mean(sig_array[cc, 0, dd, ii, jj, :, :], 1), predict_9, Artifact_length)
                    print('scale factor: '+str(ScaleFactor_9))
                    SF_array[cc, 0, dd, ii, jj] = ScaleFactor_9
                    predict_9 = predict_9*ScaleFactor_9
                    clean_9 = np.reshape(sig_9-predict_9[:len(sig_9)], (ntrials, nsamples)).T
                    clean_array[cc, 0, dd, ii, jj, :, : ] = clean_9                    
#                    # 4. evaluate the shape of kernel H0
#                    # continuely use the largest stimulus impulse 
#                    # calculate the correlation of the original signal&artifact
#                    # if the peak of the correlation results is not at idx(Winerfilterorder - 1), H0 unmatched the artifact shape, reject the data
#                    peakidx = np.argmax(predict_9[:Artifact_length])
#                    sig_peak = sig_array[cc, 0, dd, ii, jj, peakidx-12:peakidx+13, :] - sig_array[cc, 0, dd, ii, jj, peakidx-12, :]                        
#                    SigArtifact_list = []
#                    CleanArtifact_list = []
#                    for tt in range(sig_array.shape[-1]):
#                        original_signal = sig_peak[:, tt]
#                        predict_signal = predict_9[peakidx-12:peakidx+13]
#                        clean_signal = original_signal - predict_signal
#                        SigArtifact_list.append(np.argmax(np.correlate(predict_signal,original_signal, 'full')))
#                        CleanArtifact[cc, dd, ii, jj, tt] = np.argmax(np.correlate(predict_signal,clean_signal, 'full'))
#                        CleanArtifact_list.append(np.argmax(np.correlate(predict_signal,clean_signal, 'full')))
##                            use variance to evaluate how clean the signal is 
##                            x = np.correlate(predict_signal, clean_signal, 'full')
##                            x = x.astype('float64')
##                            x_var = sta.variance(x)
##                            y = np.correlate(predict_signal, original_signal, 'full')
##                            y = y.astype('float64')
##                            y_var = sta.variance(y)                           
#                    if not len(np.unique(np.array(SigArtifact_list))) == 1 and np.mean(np.array(SigArtifact_list)) == WienerFilterOrder-1:
#                        reject_mark[cc, dd, ii, jj] = 1
#                        print('H0 unmatched, reject the data')
#                        continue
#                    if not len(np.unique(np.array(CleanArtifact_list))) >= int(ntrials/3):
#                        reject_mark[cc, dd, ii, jj] = 2
#                        print('H0 unmatched, reject the data')
#                        continue                    
                    # 5. use H0 to do artifact rejection on 4500pps
                    sig_45 = concatenation(sig_array[cc, 1, dd, ii, jj, :, :])
                    stim = StimulusData[1, dd, ii, jj, :, 0]
                    stim_45 = np.matlib.repmat(stim,1,ntrials).T[:, 0]
                    stim_45[stim_45 < 0] = 0
                    predict_45 = np.convolve(H0, stim_45)
                    correlate_45 = np.correlate(sig_45[:Artifact_length], predict_45[:Artifact_length], 'full')
                    idxCorr_45 = np.argmax(correlate_45[Artifact_length-5:Artifact_length+5])
                    shift = idxCorr_45-(5-1)
                    if shift == 0:
                        predict_45 = predict_45
                    elif shift < 0:
                        predict_45 = predict_45[np.abs(shift):len(sig_45)+np.abs(shift)]
                    elif shift > 0:
                        predict_45 = np.concatenate((np.zeros((shift)), predict_45[:len(sig_45)-shift]))
                    ScaleFactor_45 = CalcSF(np.mean(sig_array[cc, 1, dd, ii, jj, :, :], 1), predict_45, Artifact_length)
                    SF_array[cc, 1, dd, ii, jj] = ScaleFactor_45
                    predict_45 = predict_45*ScaleFactor_45
                    Predict_array[cc, 1, dd, ii, jj, :Artifact_length] = predict_45[:Artifact_length]
                    clean_45 = np.reshape(sig_45-predict_45[:len(sig_45)], (ntrials, nsamples)).T
                    clean_array[cc, 1, dd, ii, jj, :, : ] = clean_45
                    # 6. calculate post artifact rejection SNR to evaluate how clean the signal is
#                    for tt in range(ntrials):
#                        fft_array[cc, :, dd, ii, jj, :int(np.around(Fs/4)), tt] = CalcFFT(clean_array[cc, :, dd, ii, jj, :, tt], Artifact_length, Fs)
#                        SNR_array[cc, 0, dd, ii, jj, tt] = CalcSNR(fft_array[cc, 0, dd, ii, jj, :], 900, Fs)
#                        SNR_array[cc, 1, dd, ii, jj, tt] = CalcSNR(fft_array[cc, 1, dd, ii, jj, :], 4500, Fs)
#                    if np.array(np.where(SNR_array[cc, 0, dd, ii, jj, :]>1.5)).shape[1] > int(ntrials/5):
#                        reject_mark[cc, dd, ii, jj] = 3
#                        print('artifact residue, reject the data')
#                        continue
#    plt.figure(figsize=(10,15))
#    plt.xticks(np.arange(1, 33, 1))
#    plt.title('reject mark')
#    #plt.subplot(1,2,1)
#    y = 0
#    for dd in range(3):
#        for ii in range(3):
#            for jj in range(3):
#                y = y+1
#                for cc in range(32):
#                    if reject_mark[cc, dd, ii, jj] == 0:                        
#                        plt.plot(cc+1,y, 'ko')
#                    elif reject_mark[cc, dd, ii, jj] == 1:
#                        plt.plot(cc+1,y, 'ro')
#                    elif reject_mark[cc, dd, ii, jj] == 2:
#                        plt.plot(cc+1,y, 'bo')
#                    else:
#                        plt.plot(cc+1,y, 'go')
#    plt.savefig(results_path+sig_name[:-4]+'_reject_mark')
#    plt.close('all')
#    np.save(results_path+sig_name[:-4]+'_CleanArtifactCorrPeak.npy',CleanArtifact)
#    np.save(results_path+sig_name[:-4]+'_RejectMark.npy',reject_mark)
#    print(nsamples/Fs)
    np.save(results_path+sig_name[:-4]+'_H0.npy',H0_array)
    np.save(results_path+sig_name[:-4]+'_CleanSig.npy',clean_array)
#    np.save(results_path+sig_name[:-4]+'_SNR.npy',SNR_array)
    np.save(results_path+sig_name[:-4]+'_ScaleFactor.npy',SF_array)
#    np.save(results_path+sig_name[:-4]+'_FFT.npy',fft_array)
    np.save(results_path+sig_name[:-4]+'_Predict.npy',Predict_array)    
#    ArtifactRejection_dic = {'reject_mark': reject_mark, 'H0_array': H0_array, 'clean_array': clean_array, 'SNR_array': SNR_array, 'SF_array': SF_array, 'fft_array': fft_array, 'Fs': Fs, 'sig_array': sig_array}
#    np.save(results_path+sig_name[:-4]+'_ArtifactRejected.npy', ArtifactRejection_dic)
