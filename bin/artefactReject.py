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
from matplotlib import pyplot as plt

swps, stimPar=ep.readEphysFile('C:\\Users\\User\\Documents\\GitHub\\ENVvsPT_Ephys_Analysis\\CI2224_1_ITD_ENV_vs_PT_CI_1.ephys')
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
             
artefactAlign(stimPar, swps)
# artefactAlign(stimPar, swps)            
artefactReject(stimPar, swps)             
plt.clf()
plt.plot(swpBak[100].signal[:4000,10])
plt.plot(swps[100].signal[:4000,10])  # far from perfect