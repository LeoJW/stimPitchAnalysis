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
# import seaborn as sns
from scipy.signal import find_peaks
from scipy.stats import binned_statistic
from pyfuncs import *





#%% Start with single individual
date = '20210721'

# Channel names to process
channelsEMG = ['LDVM','LDLM','RDLM','RDVM']
channelsExtra = ['stim']
channelsFT = ['fx','fy','fz','mx','my','mz']

# Plot controls
wbBefore = 4
wbAfter = 4

# Filter Controls
hpfCutoff = 70
lpfCutoff = 500

# Wingbeat finding
zforceCutoff = 40
wbdistance = 200
fzheight = 0.05

# Thresholds
stimthresh = 3 # threshold to count stim channel as "on"


# Manually entered delays
# TODO: Add spike-sorting, automatically find delay PER WINGBEAT
# delay = [3, 10, 8, 2, 14, 1, 6, 4, 15, 29, 11, 20]
delay = [10, 7, 15, 18, 22, 8, 3, 28, 12, 17, 9, 14, 21, 4, 4, 13, 12, 17, 19, 21, 25, 25, 11, 8, 6, 10, 9]


#- Load data
# Read empty FT for bias
biasdata, colnames, fsamp = readMatFile(date, 'empty', doFT=True)
bias = np.zeros((6,1))
for i in range(6):
    bias[i] = np.mean(biasdata[colnames[i+1]])
# Read program guide to find good trials with delay
goodTrials = whichTrials(date)
# Loop over all, read in data
for iabs, i in enumerate(np.arange(goodTrials[0], goodTrials[1]+1)):
    # Make string version of trial
    trial = str(i).zfill(3)
    # print to let user know what's up
    print('  '+trial)
    
    # Read data 
    emg, emgnames, _ = readMatFile(date, trial, doFT=False, grabOnly=channelsEMG+channelsExtra)
    ftd, ftdnames, _ = readMatFile(date, trial, doFT=True, bias=bias)
    
    # Filter data
    for name in channelsEMG: # Filter EMG
        emg[name] = butterfilt(emg[name], hpfCutoff, fsamp, order=4, bandtype='high')
    for name in ftdnames: # Filter FT
        ftd[name] = butterfilt(ftd[name], lpfCutoff, fsamp, order=4, bandtype='low')
        
    # Put everything together into a dataframe
    dtemp = pd.DataFrame({**emg, **ftd})
    
    # Grab wingbeats from filtered z force
    wb = find_peaks(butterfilt(dtemp['fz'], zforceCutoff, fsamp, order=4, bandtype='lowpass'),
                    distance=wbdistance,
                    height=fzheight)[0]
    # Make long-form wingbeat column in dataframe (useful for stuff)
    dtemp['wb'] = 0
    dtemp.loc[wb, 'wb'] = 1
    dtemp['wb'] = np.cumsum(dtemp['wb'])
    
    # Make delay column
    dtemp['delay'] = delay[iabs]
    
    # Make trial column
    dtemp['trial'] = i
    
    # Create phase vector by rescaling time based on wingbeats
    dtemp['phase'] = np.nan
    for j in np.arange(len(wb)-1):
        ind = np.arange(wb[j], wb[j+1])
        dtemp.loc[ind, 'phase'] = np.linspace(0, 1, len(ind))
        
    # Get stim indices
    si = np.where(np.logical_and(dtemp['stim']>3,
                                 np.roll(dtemp['stim']<3, 1)))[0]
    

    # Waste memory and create second phase column to count multiple wingbeats 
    # (0->1, 1->2, etc rather than 0->1, 0->1)
    dtemp['multphase'] = dtemp['phase']
    dtemp['wbstate'] = 'regular'
    dtemp['pulse'] = np.nan
    # Label wingbeats as being pre-, during, or post- stimulation
    for s in si:
        # get which wingbeat this stim is on
        stimwb = dtemp.loc[s, 'wb']
        
        #--- Pre-stim wingbeats
        # Grab indices of this pre-stim period
        inds = np.where(np.logical_and(dtemp['wb']>=(stimwb-wbBefore),
                                       dtemp['wb']<stimwb))[0]
        # Label as pre-stim period
        dtemp.loc[inds, 'wbstate'] = 'pre'
        dtemp.loc[inds, 'pulse'] = s
        # Change phase column to go from 0->x where x is number of "before" wingbeats
        for ii in np.arange(wbBefore):
            thiswb = stimwb - wbBefore + ii        
            dtemp.loc[dtemp['wb']==thiswb, 'multphase'] += ii
                
        #--- Stim wingbeat
        # label stim wingbeat
        dtemp.loc[dtemp['wb']==stimwb, 'wbstate'] = 'stim'
        dtemp.loc[dtemp['wb']==stimwb, 'pulse'] = s
        
        #--- Post-stim wingbeats
        inds = np.where(np.logical_and(dtemp['wb']<=(stimwb+wbAfter),
                                       dtemp['wb']>stimwb))[0]
        # Label as post-stim period
        dtemp.loc[inds, 'wbstate'] = 'post'
        dtemp.loc[inds, 'pulse'] = s 
        # TODO: check if this ^ can be optimized. Can do all 3 in single line if needed
        # Set up phase so it goes from 0->x where x is number of "before" wingbeats
        for ii in np.arange(wbAfter):
            thiswb = stimwb + ii + 1
            dtemp.loc[dtemp['wb']==thiswb, 'multphase'] += ii

    # Remove regular wingbeats
    # dtemp = dtemp[dtemp['wbstate'].isin(['pre','stim','post'])]
    
    # Clean EMG channels to NAN during stimulation period (to remove artifact)
    dtemp.loc[dtemp['stim']>stimthresh, channelsEMG] = np.nan
    
    # Add to full dataframe
    if i==goodTrials[0]:
        df = dtemp
    else:
        df = pd.concat([df,dtemp])

da = df.copy()
# Version without regular wingbeats
df = df[df['wbstate'].isin(['pre','stim','post'])]
        

#%%

binPlot(df.loc[df['delay']<20, ],
        plotvars=['fz','mx'],
        groupvars=['wbstate','delay'],
        colorvar='delay',
        numbins=300, wbBefore=wbBefore, wbAfter=wbAfter,
        doSTD=False)



#%%

# Look at all traces for single variable (mx) for single set of delays
binPlot(df.loc[df['delay']==8 , ],
        plotvars=['fz','mx'],
        groupvars=['wbstate','delay','wb'],
        colorvar='wb',
        numbins=300, wbBefore=wbBefore, wbAfter=wbAfter,
        doSTD=False,
        doSummaryStat=False)



# quickPlot(date, '009',
#           tstart=10, tend=15,
#           plotnames=['LDVM', 'LDLM','RDLM','RDVM','stim'])



#%%
trial = 13


# Make aggregate control dictionary
aggdict = {}
for i in list(df.select_dtypes(include=np.number)): # loop over all numeric columns
    aggdict[i] = 'mean'
aggdict['wbstate'] = 'first'

bob = da.loc[da['trial']==trial,].groupby(['wb','trial']).agg(aggdict)


# Plot wb mean values for this trial
fig, ax = plt.subplots(len(channelsFT), 1)
for i, varname in enumerate(channelsFT):
    ax[i].plot(bob['Time'], bob[varname], marker='.')
# Replot stimulus wingbeats as red
for i, varname in enumerate(channelsFT):
    ax[i].plot(bob.loc[bob['wbstate']=='stim', 'Time'],
               bob.loc[bob['wbstate']=='stim', varname],
               'r.')



#%%

# Make aggregate control dictionary
aggdict = {}
for i in list(df.select_dtypes(include=np.number)): # loop over all numeric columns
    aggdict[i] = 'mean'
aggdict['wbstate'] = 'first'


# Figure setup
fig, ax = plt.subplots(len(channelsFT), 1)
statenames = np.unique(df['wbstate'])
colormax = np.max(df['delay'])

# Create aggregated dataframe
dt = df.groupby(['wb','trial']).agg(aggdict)
# Loop over pre, stim, post
for j,state in enumerate(statenames):
    # Loop over plot variables
    for i, varname in enumerate(channelsFT):
        data = dt.loc[dt['wbstate']==state, ]
        ax[i].plot(np.ones(len(data)) + j,
                   data[varname],
                   '.',
                   color = viridis(group['delay'].iloc[0]/colormax)[0:3])




'''
Bug fixes
- Long pauses between wingbeats get counted as single wingbeats. Need to remove those pauses
- Some traces (delay==4) grab more wingbeats than wbBefore requests (5 instead of 4)
'''
