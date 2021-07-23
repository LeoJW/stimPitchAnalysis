#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 12:06:44 2021

Stimulation experiment analysis script

@author: leo
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import binned_statistic
from pyfuncs import *





#%% Start with single individual
date = '20210708'

# Channel names to process
channelsEMG = ['LDVM','LDLM','RDLM','RDVM']
channelsExtra = ['stim']
channelsFT = ['fx','fy','fz','mx','my','mz']

# Plot controls
wbBefore = 2
wbAfter = 2

# Filter Controls
hpfCutoff = 70
lpfCutoff = 500
# Wingbeat finding
zforceCutoff = 40
wbdistance = 200


#- Load data
# Read empty FT for bias
biasdata, colnames, fsamp = readMatFile(date, 'empty', doFT=True)
bias = np.zeros((6,1))
for i in range(6):
    bias[i] = np.mean(biasdata[colnames[i+1]])
# Read program guide to find good trials with delay
goodTrials = whichTrials(date)
# Loop over all, read in data
# for i in np.arange(goodTrials[0], goodTrials[1]+1):
for i in [6]:
    trial = str(i).zfill(3)
    
    # Read data 
    emg, emgnames, _ = readMatFile(date, trial, doFT=False, grabOnly=channelsEMG+channelsExtra)
    ftd, ftdnames, _ = readMatFile(date, trial, doFT=True, bias=bias)
    
    # Filter data
    for name in channelsEMG: # Filter EMG
        emg[name] = butterfilt(emg[name], hpfCutoff, fsamp, order=4, bandtype='high')
    for name in ftdnames: # Filter FT
        ftd[name] = butterfilt(ftd[name], lpfCutoff, fsamp, order=4, bandtype='low')
        
    # Put everything together into a dataframe
    df = pd.DataFrame({**emg, **ftd})
    
    # Grab wingbeats from filtered z force
    wb = find_peaks(butterfilt(df['fz'], zforceCutoff, fsamp, order=4, bandtype='lowpass'),
                    distance=wbdistance)[0]
    # Make long-form wingbeat column in dataframe (useful for stuff)
    df['wb'] = 0
    df.loc[wb, 'wb'] = 1
    df['wb'] = np.cumsum(df['wb'])
    
    # Get stim indices
    si = np.where(np.logical_and(df['stim']>3,
                                 np.roll(df['stim']<3, 1)))[0]
    
    # Create phase vector by rescaling time based on wingbeats
    df['phase'] = np.nan
    for j in np.arange(len(wb)-1):
        ind = np.arange(wb[j], wb[j+1])
        df.loc[ind, 'phase'] = np.linspace(0, 1, len(ind))
    
    
    yvar = ftd['mx']    
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    
    # Fit mean x wb pre-, during stim, and x wb after for all channels
    xpre = []
    xstim = []
    xpost = []
    
    for s in si:
        # get which wingbeat this stim is on
        stimwb = df.loc[s, 'wb']
        
        #--- Pre-stim wingbeats
        # inds = np.where(np.logical_and(wbvec>=(stimwb-wbBefore), wbvec<stimwb))[0]
        inds = np.where(np.logical_and(df['wb']>=(stimwb-wbBefore), df['wb']<stimwb))[0]
        # xvec = phase[inds]
        xvec = df.loc[inds, 'phase']
        # Set up phase so it goes from 0->x where x is number of "before" wingbeats
        for ii in np.arange(wbBefore):
            thiswb = stimwb - wbBefore + ii
            # thiswbinds = np.where(wbvec[inds]==thiswb)[0]
            thiswbinds = np.where(df.loc[ind, 'wb']==thiswb)[0]
            xvec[thiswbinds] = xvec[thiswbinds] + ii
        # Plot
        ax[0].plot(xvec, yvar[inds], lw=0.5)
        
        
        #--- Stim wingbeat
        # inds = np.where(wbvec==stimwb)[0]
        inds = np.where(df['wb']==stimwb)[0]
        # xvec = phase[inds]
        xvec = df.loc[inds, 'phase']
        # Plot
        ax[1].plot(xvec, yvar[inds], lw=0.5)
        
        #--- Post-stim wingbeats
        # inds = np.where(np.logical_and(wbvec<=(stimwb+wbAfter), wbvec>stimwb))[0]
        inds = np.where(np.logical_and(df['wb']<=(stimwb+wbAfter), df['wb']>stimwb))[0]
        # xvec = phase[inds]
        xvec = df.loc[inds, 'phase']
        # Set up phase so it goes from 0->x where x is number of "before" wingbeats
        for ii in np.arange(wbAfter):
            thiswb = stimwb + ii + 1
            # thiswbinds = np.where(wbvec[inds]==thiswb)[0]
            thiswbinds = np.where(df.loc[ind, 'wb']==thiswb)[0]
            xvec[thiswbinds] = xvec[thiswbinds] + ii
        # Plot
        ax[2].plot(xvec, yvar[inds], lw=0.5)
    

    
'''
TODO:
    - Come up with way to create mean lines (binned means? LOESS?)
    - Way to store information from all trials
    - data structure to generalize to many individuals and trials
'''





