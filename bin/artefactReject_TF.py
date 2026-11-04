# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sys import path
path.append('/home/colliculus/behaviourBoxes/software/ratCageProgramsV3/')
import RZ2ephys as ep
# import os
import numpy as np
from scipy.signal import correlate2d, correlate, convolve, butter, filtfilt, iirnotch, resample_poly, csd, windows, welch, get_window, resample
from matplotlib import pyplot as plt

swps, stimPar=ep.readEphysFile('C:\\Users\\User\\Documents\\GitHub\\ENVvsPT_Ephys_Analysis\\CI2224_1_ITD_ENV_vs_PT_CI_1.ephys')

# swps, stimPar=ep.readEphysFile('Q:/HNO-F-NeuroBio-EP-Daten/ENVvsPT/ENVvsPT_raw_data/ITD+/CI2224_ITD_ENV_vs_PT_CI_1/CI2224_1_ITD_ENV_vs_PT_CI_1.ephys') # This is for Freiburg Code

#%%
def nonUniqueParams(stimPar):
    nup={}
    for par in stimPar.keys():
        if par[:4]=='imp_':
            continue
        if par=='timeStamp':
            continue
        # print(par)
        upar=np.sort(stimPar[par].unique())
        if len(upar)>1:
            nup[par]=upar
    return nup

# def paramCombos(stimPar,nup)


def myvstack(Y,newRow):
    if len(Y)==0:
        Y=np.array([newRow])
    else:
        if np.size(newRow)==0:
            return Y
        else:
            Y=np.vstack((Y,[newRow]))  
    return Y

def permute(A,B):
    Y=np.array([])
    A=np.array(A)
    B=np.array(B) 
    if A.size==0:
        return np.array([])
    for a in A:
        a=np.array(a)
        if B.size==0:
            Y=myvstack(Y,a)
        else:
            for b in B:
                b=np.array(b)
                newRow=np.append(a,b)
                Y=myvstack(Y,newRow)
    return Y

def permutations(nup):
    p=[]   
    for par in nup.keys(): 
        p=permute(nup[par],p)
    return(np.fliplr(p))



def sweepIndecesFor(stimPar, parNames, parValues):
    stillTrue=np.ones(len(stimPar))
    for ii in range(len(parNames)):
        stillTrue=np.logical_and(stillTrue,stimPar[parNames[ii]]==parValues[ii])
    return(np.where(stillTrue)[0])
      
# swpIdx=sweepIndecesFor(stimPar, parNames, p[1,:])

# plt.clf()
# for idx in swpIdx:
#     plt.plot(swps[idx+1].signal[:1000,10])


def meanSweepsOverIdx(swps,swpIdx, nSamples=6000):
    # idx+=1 # remember that stim index 0 corresponds to swps index
    aSig=swps[swpIdx[0]+1].signal[:nSamples,:].copy()
    aSig-=aSig[0,:]
    for ii in range(1,len(swpIdx)):
        epoch=swps[swpIdx[ii]+1].signal[:nSamples,:].copy()
        epoch-=epoch[0,:]
        aSig+=epoch
    return aSig / len(swpIdx)

# meanSig=meanSweepsOverIdx(swps,swpIdx)
# plt.clf()
# plt.plot(meanSig[:,10])

swpBak=[]
for swp in swps:
    swpBak.append(swp.copy())
    


#Import AMUA and filter types- same as we used previously:
fs_down = 2000    

def AMUAFilterCoeffs(fs,lowpass=6000):
        nyq = 0.5*fs
        bBand,aBand = butter(2,(300/nyq, 6000/nyq),'bandpass')  
        bLow,aLow = butter(2,(lowpass/nyq),'lowpass')
        bNotch, aNotch = iirnotch(50, 30, fs)
        return [[bBand, aBand], [bLow, aLow], [bNotch, aNotch]]
          
     
def calcAMUA(fs, ori_signal, Fs_downsample, padLen=300):
        coefs=AMUAFilterCoeffs(fs)
        bpCoefs=coefs[0]
        lpCoefs=coefs[1]
        NotchCoefs = coefs[2]
        insig = filtfilt(NotchCoefs[0], NotchCoefs[1], ori_signal, axis=0, padlen=padLen)
        insig = np.flip(insig)
        insig = filtfilt(bpCoefs[0],bpCoefs[1], insig, axis=0, padlen=padLen)
        insig = np.abs(insig)
        insig = filtfilt(lpCoefs[0],lpCoefs[1],insig,axis=0, padlen=padLen)
        insig = np.flip(insig)          
        # Fs_downsample
        # signal = resample_poly(insig, Fs_downsample, int(fs), axis=0)
        downsample_length = int((insig.shape[0]/fs)*Fs_downsample)
        signal=resample(insig,downsample_length)
        
        return signal    

    
#%%    
# swps=[]
# for swp in swpBak:
#     swps.append(swp.copy())
    
#%%

def artefactAlign(stimPar, swps):
    nup=nonUniqueParams(stimPar)
    parNames=list(nup.keys())
    p=permutations(nup)
    for row in range(p.shape[0]):
        # cross-correlate to check alignment
        swpIdx=sweepIndecesFor(stimPar, parNames, p[row,:])
        stimSamples = int(swps[0].sampleRate*stimPar['duration (s)'][swpIdx[0]]) 
        meanSig=meanSweepsOverIdx(swps,swpIdx,nSamples=stimSamples).mean(axis = 1)
        for rr, idx in enumerate(swpIdx):               
            xcorr=np.correlate(meanSig,swps[idx+1].signal[:stimSamples,:].mean(axis = 1),'full')
            peakIdxRef = np.argmax(xcorr) 
            requiredShift=peakIdxRef-stimSamples+1
            if requiredShift !=0 :                     
                print(row, idx, requiredShift)
                # break
            if requiredShift < 0:
                # sweep idx+1 was triggered requiredShift samples too late
                swps[idx].signal=np.concatenate((swps[idx].signal, swps[idx+1].signal[:-requiredShift,:]))
                swps[idx+1].signal=swps[idx+1].signal[-requiredShift:,:]
                # break
            if requiredShift > 0:
                swps[idx].signal = swps[idx].signal[:-requiredShift, :]
                swps[idx+1].signal = np.concatenate((swps[idx].signal[-requiredShift:, :], swps[idx+1].signal))
                # break

def artefactReject(stimPar, swps):
    nup=nonUniqueParams(stimPar)
    parNames=list(nup.keys())
    p=permutations(nup)
    for row in range(p.shape[0]):
        swpIdx=sweepIndecesFor(stimPar, parNames, p[row,:])
        stimSamples = int(swps[0].sampleRate*stimPar['duration (s)'][swpIdx[0]]) +50
        meanSig=meanSweepsOverIdx(swps,swpIdx,nSamples=stimSamples)
        for idx in swpIdx:
            swps[idx+1].signal[:stimSamples,:]-=meanSig


# The new function I wrote --> I dont know exactly what to do in the end with the amua arrays, however they are calculated and can be used further :-) 
def calculateAMUAandSpikes (stimPar, swps):
    sampleRate = swps[0].sampleRate
    AMUAPen = [] # list that will be column with AMUA values
    for swp in range(len(swps)):           
        if swp == 0:
            continue                         #######This is to get rid of the first uninformative test sweep --> If you have any better idea please implement it! :-) #######    
        response_start = 0.005 
        response_end = 0.055
        baseline_start = 0.325
        
        tempSwp = swps[swp].spikeFilter()                  # take the sweep and filtering for frequency range of spikes --> .spikeFilter in Rzephys Butter filter 300-3000Hz 
        thcrossings = tempSwp.thresholdCrossings()          # get spikes - dictonary with array of crossings for each chan - careful!!! first to keays of dictionary are something else!!!
        spikeCountSwp = []                                  #Count the spike threshold crossing for each channel
        AMUASwp_Area = []
                                                                            # Loop through the channels (32) --> thcrossings[1] = all the thresholdcrossings in the exact time (up to tLim) 
        for chan in range(swps[swp].signal.shape[1]):
            spikeCountChan = np.sum((thcrossings[chan] >= response_start) & (thcrossings[chan] <= response_end))     # check how many spikes in analysing window (stim duration + 50ms)
            spikeCountSwp.append(spikeCountChan)                                                                        # Check this also to have another way of analysing --> Done in the eLife Paper 2021                
            #AMUA
            amuaSignal = calcAMUA(sampleRate, swps[swp].signal[int(response_start * sampleRate):int(response_end * sampleRate),chan], fs_down) # now take the sweep and calculate AMUA in the response window only
            amuaBaseline = calcAMUA(sampleRate, swps[swp].signal[int(baseline_start * sampleRate):,chan], fs_down)                              # now take the sweep and calculate AMUA in the baseline window only
            amuaTotal = calcAMUA(sampleRate, swps[swp].signal[:,chan], fs_down)                                  # now take the sweep and calculate AMUA in the WHOLE SWEEP
            
            amuaArea = round(sum(amuaSignal)*sampleRate, 2)                   # calculating the area under the amua signal to have a quantitative Value for the response, done in the eLife Paper 2021 --> Optional
            AMUASwp_Area.append(amuaArea)                                       # Save the Value from each are under the curve for each Channel seperately. 
        AMUAPen.append(AMUASwp_Area)
        
             
artefactAlign(stimPar, swps)
# artefactAlign(stimPar, swps)            
artefactReject(stimPar, swps)

calculateAMUAandSpikes(stimPar, swps)

plt.clf()
plt.plot(swpBak[100].signal[:4000,10])
plt.plot(swps[100].signal[:4000,10])  # far from perfect




















