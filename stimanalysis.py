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



#%% whichTrials improvement
def whichTrials(date, purpose='good'):
    # Move to data directory
    startdir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(os.path.join(startdir, os.pardir + '/' + date))
    # Check if programGuide exists
    contents = os.listdir()
    guide = [s for s in contents if 'program' in s]
    # yell if it doesn't 
    if len(guide)==0:
        print('No programGuide file for this moth! Make one plz')
        return
    # read file
    names = []
    table = []
    reader = csv.reader(open(guide[0]))
    for row, record in enumerate(reader):
        names.append(record[0])
        table.append([int(s) for i,s in enumerate(record) if i!=0 and s!=''])
    # if looking for good delay trial
    if purpose=='good':
        # Grab first good delay trial
        start = table[2][0]
        # grab last good delay trial
        if len(table[3])==0:
            end = table[2][1]
        else:
            end = table[3][1]
        # Create range
        trials = np.arange(start, end+1)
        # Remove any characterization that may have happened in middle
        if len(table[1])>2:
            # Loop over how many periods may have happened
            for i in np.arange(2, len(table[1]), 2):                
                trials = [x for x in trials if x<table[1][i] or x>table[1][i+1]]
        return trials


trials = whichTrials('20210727')



#%% Start with single individual
date = '20210727'
# date = '20210708'

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
# delay = [10, 7, 15, 18, 22, 8, 3, 28, 12, 17, 9, 14, 21, 4, 4, 13, 12, 17, 19, 21, 25, 25, 11, 8, 6, 10, 9]
delay = [9, 2, 18, 16, 9, 12, 5, 18, 16, 20, 24, 2, 5, 21]


#- Load data
# Read empty FT for bias
biasdata, colnames, fsamp = readMatFile(date, 'empty', doFT=True)
bias = np.zeros((6,1))
for i in range(6):
    bias[i] = np.mean(biasdata[colnames[i+1]])
# Read program guide to find good trials with delay
goodTrials = whichTrials(date)

pulsecount = 1
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
    dtemp['wbrel'] = dtemp['wb']
    dtemp['pulse'] = 0
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
        dtemp.loc[inds, 'pulse'] = pulsecount
        # Change phase column to go from 0->x where x is number of "before" wingbeats
        for ii in np.arange(wbBefore):
            thiswb = stimwb - wbBefore + ii        
            dtemp.loc[dtemp['wb']==thiswb, 'multphase'] += ii
            dtemp.loc[dtemp['wb']==thiswb, 'wbrel'] -= stimwb
                
        #--- Stim wingbeat
        # label stim wingbeat
        dtemp.loc[dtemp['wb']==stimwb, 'wbstate'] = 'stim'
        dtemp.loc[dtemp['wb']==stimwb, 'wbrel'] = 0
        dtemp.loc[dtemp['wb']==stimwb, 'pulse'] = pulsecount
        
        #--- Post-stim wingbeats
        inds = np.where(np.logical_and(dtemp['wb']<=(stimwb+wbAfter),
                                       dtemp['wb']>stimwb))[0]
        # Label as post-stim period
        dtemp.loc[inds, 'wbstate'] = 'post'
        dtemp.loc[inds, 'pulse'] = pulsecount
        # TODO: check if this ^ can be optimized. Can do all 3 in single line if needed
        # Set up phase so it goes from 0->x where x is number of "before" wingbeats
        for ii in np.arange(wbAfter):
            thiswb = stimwb + ii + 1
            dtemp.loc[dtemp['wb']==thiswb, 'multphase'] += ii
            dtemp.loc[dtemp['wb']==thiswb, 'wbrel'] -= stimwb
        
        # Increment global pulse counter
        pulsecount += 1
        
        
    
    # Clean EMG channels to NAN during stimulation period (to remove artifact)
    for name in channelsEMG:
        dtemp.loc[dtemp['stim']>stimthresh, name] = np.nan
    
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
binPlot(df.loc[df['delay']==18, ],
        plotvars=['mx'],
        groupvars=['wbstate','delay','wb'],
        colorvar='wb',
        numbins=300, wbBefore=wbBefore, wbAfter=wbAfter,
        doSTD=False,
        doSummaryStat=False)


# df.loc[np.logical_and(df['delay']==21, df['wbstate']=='stim'), 'mx'].agg(['mean','std','min','max'])




quickPlot(date, '003',
          tstart=1, tend=15,
          plotnames=['RDVM', 'RDLM','LDLM','LDVM','mx'])



#%% Difference between traces 1wb pre, during, post stim





#%%
trial = 9


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


# # Figure setup
# fig, ax = plt.subplots(len(channelsFT), 1)
# statenames = np.unique(df['wbstate'])
# viridis = cmx.get_cmap('viridis')
# colormax = np.max(df['delay'])

# # Create aggregated dataframe
# dt = df.groupby(['wb','trial']).agg(aggdict)
# count = 0
# for name, group in dt.groupby('delay'):
#     # Loop over pre, stim, post
#     for j,state in enumerate(statenames):
#         # Loop over plot variables
#         for i, varname in enumerate(channelsFT):
#             data = group.loc[group['wbstate']==state, ]
#             ax[i].plot(np.ones(len(data)) + j + 0.05*np.random.rand(len(data)) + count,
#                        data[varname],
#                        '.',
#                        alpha=0.5,
#                        color = viridis(data['delay'].iloc[0]/colormax)[0:3])

#     count += 0.05

import seaborn as sns

# Create aggregated dataframe
dt = df.groupby(['wb','trial']).agg(aggdict)

dt = dt.loc[dt['wbrel']>-5,]

sns.catplot(data=dt, 
            kind="boxen",
            x='wbrel',
            y='mx',
            hue='delay',
            palette='viridis')



#%% Pre, during, post differences

dt = df.groupby(['wb','trial']).agg(aggdict)
dt = dt.loc[dt['wbrel']>-5,]
# Loop over non-numeric columns that aren't wb
for name in [s for s in list(dt.select_dtypes(include=np.number)) 
             if s not in 'wb' 
             and s not in 'wbrel'
             and s not in 'delay']:
    dt[name] = dt[name] - np.roll(dt[name], 1)
    
# Keep only 1 wb pre, during, and post stim
dt = dt.loc[np.logical_and(dt['wbrel']>=-1, dt['wbrel'] <=1), ]
    
# sns.catplot(data=dt, 
#             kind="boxen",
#             x='wbrel',
#             y='mx',
#             hue='delay',
#             palette='viridis')



'''
Bug fixes
- Long pauses between wingbeats get counted as single wingbeats. Need to remove those pauses
- Some traces (delay==4) grab more wingbeats than wbBefore requests (5 instead of 4)
'''
