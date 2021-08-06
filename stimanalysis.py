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

import time as systime



#%% Start with single individual
date = '20210803_1'

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


#- Load data
# Read empty FT for bias
biasdata, colnames, fsamp = readMatFile(date, 'empty', doFT=True)
bias = np.zeros((6,1))
for i in range(6):
    bias[i] = np.mean(biasdata[colnames[i+1]])
# Read program guide to find good trials with delay
goodTrials = whichTrials(date)

# Create variables for loop
pulsecount = 1
stiminds = [[] for x in range(goodTrials[-1])]

# Loop over all, read in data
for i in goodTrials:
    tic = systime.perf_counter()
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
    stiminds[i-1] = si

    # Waste memory and create second phase column to count multiple wingbeats 
    # (0->1, 1->2, etc rather than 0->1, 0->1)
    dtemp['multphase'] = dtemp['phase']
    dtemp['wbstate'] = 'regular'
    dtemp['wbrel'] = dtemp['wb']
    dtemp['pulse'] = 0
    dtemp['stimdelay'] = 0
    dtemp['stimphase'] = 0
    # Label wingbeats as being pre-, during, or post- stimulation
    for s in si:
        # get which wingbeat this stim is on
        stimwb = dtemp.loc[s, 'wb']
        
        # Save the stim delay and phase
        stimdelay = (s - np.argmax(dtemp['wb']==stimwb))/fsamp*1000 #in ms
        stimphase = stimdelay*fsamp/1000/len(dtemp.loc[dtemp['wb']==stimwb])
        
        #--- Pre-stim wingbeats
        # Grab indices of this pre-stim period
        inds = np.where(
            np.logical_and(
                np.logical_and((dtemp['wb']>=(stimwb-wbBefore)), (dtemp['wb']<stimwb)),
                (dtemp['wbstate']!='stim')))[0]
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
        inds = dtemp['wb']==stimwb
        dtemp.loc[inds, 'wbstate'] = 'stim'
        dtemp.loc[inds, 'wbrel'] = 0
        dtemp.loc[inds, 'pulse'] = pulsecount
        dtemp.loc[inds, 'stimdelay'] = stimdelay
        dtemp.loc[inds, 'stimphase'] = stimphase
        
        #--- Post-stim wingbeats
        inds = np.where(
            np.logical_and(
                np.logical_and((dtemp['wb']<=(stimwb+wbAfter)), (dtemp['wb']>stimwb)),
                (dtemp['wbstate']!='stim')))[0]
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
    print(systime.perf_counter()-tic)
        

da = df.copy()
# Version without regular wingbeats
df = df[df['wbstate'].isin(['pre','stim','post'])]
        

#%% Pull in spike times from spike sorting

# Constants
stimwindow = 0.001


def readSpikeSort(date, muscles=['LDVM','LDLM','RDLM','RDVM'],
                  stimAmplitudeThresh=4):
    # Jump out to spikesort dir
    startdir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(os.path.join(startdir, os.pardir, 'spikesort'))
    # Jump into dir for this date
    os.chdir(date)
    # Get file names in this dir, ignoring notes
    filenames = [s for s in os.listdir() if s != 'note']
    
    # Prepare storage variables
    spikes = {}
    waveforms = {}
    for m in muscles:
        spikes[m] = []
        spikes[m + '_sorttype'] = []
        waveforms[m] = []
    # Loop over muscles
    for m in muscles:
        # Find files for this muscle
        mfiles = [s for s in filenames
                  if m in s
                  if 'sort' in s]
        # If no files for this muscle, yell and continue
        if len(mfiles) == 0:
            print(m + ' has no sorted files!')
            continue
        # Loop over all sorted files 
        # (usually 1, may be more if _up or _down variant)
        for sortfile in mfiles:
            # Determine type of sort (up, down, or regular)
            if 'up' in sortfile:
                thistype = 'up'
            elif 'down' in sortfile:
                thistype = 'down'
            else:
                thistype = 'reg'
            # Read in file
            mat = scipy.io.loadmat(sortfile)
            # Loop over "channels" (actually just trials) and grab data
            for ch in [s for s in list(mat) if '__' not in s]:
                # grab channel/trial number 
                chnumber = int(ch[-2:])
                # grab spike times and put together with trial number
                temparray = np.column_stack((mat[ch][:,1],
                                             chnumber*np.ones(len(mat[ch][:,1]))))
                # Remove any obvious stim artifacts (high amplitude!)
                rminds = np.where(np.any(mat[ch][:,2:] > stimAmplitudeThresh, axis=1))[0]
                temparray = np.delete(temparray, (rminds), axis=0)
                mat[ch] = np.delete(mat[ch], (rminds), axis=0)
                
                # save spike times
                spikes[m].append(temparray)
                # Save sort type
                spikes[m + '_sorttype'].append(thistype)
                # save waveforms
                waveforms[m].append(mat[ch][:,2:])
                
        # Do a last vstack of all the nparrays in a list
        spikes[m] = np.vstack(spikes[m])
        waveforms[m] = np.vstack(waveforms[m])
    return spikes, waveforms
                

# Load spike times for this date
spikes, waveforms = readSpikeSort(date)


closespikes = {}
susrows = {}
for m in channelsEMG:
    closespikes[m] = []
    susrows[m] = []

for m in channelsEMG:
    # Loop over trials that are also in goodtrials
    for i in list(set(np.unique(spikes[m][:,1])).intersection(goodTrials)):
        #--- Flip spike times to work on the -len : 0 time scale
        # Get time length of this trial
        tlength = np.min(da.loc[da['trial']==i, 'Time'])
        # Get inds of this trial for spikes
        inds = spikes[m][:,1]==i
        spikes[m][inds,0] = spikes[m][inds,0] + tlength
        
        #--- Remove spike times that fall within threshold of stim pulse
        stimtimes = da.loc[da['trial']==i, 'Time'][stiminds[i]].to_numpy()
        closest = np.ones(len(stimtimes), dtype=int)
        for j in range(len(stimtimes)):
            spikeDistance = abs(spikes[m][inds,0] - stimtimes[j])
            closestSpike = np.argmin(spikeDistance)
            # If none actually met condition, change value to reflect that
            if spikeDistance[closestSpike] > stimwindow:
                closest[j] = -1
            # Otherwise save spike that was closest
            else:
                closest[j] = np.where(inds)[0][closestSpike]                
            
        closespikes[m].extend(closest[closest != -1].tolist())
        
        
    # As a check: Plot all trials meeting test condiiton for being stim spikes
    plt.figure()
    susrows[m].extend(np.where(np.min(waveforms[m], axis=1) < -0.4)[0])
    for j in susrows[m]:
        plt.plot(waveforms[m][j,:])
    plt.title(m)
    
    # Other check: Plot waveforms meeting closeness condition for being stim
    # Remove empties
    closespikes[m] = [x for x in closespikes[m] if x != []]
    plt.figure()
    for j in closespikes[m]:
        plt.plot(waveforms[m][j,:])
    plt.title(m)
    
    
    


# Test: plot distribution of spikes around stimulus


#%% Test: plot random snippet with spikes marked

da['stim'] = 0
for i in 
da['stim'][]

pdata = da.loc[(da['trial']==10) & (da['Time']>-10) & (da['Time']<-8), ]

plt.figure()
plt.plot(pdata['Time'], pdata['LDVM'])




#%% Plot distribution of spike phase for each muscle (for first spike in wingbeat)


#%% Get & plot relative spike times before, during, and after stimulus

#%% Plot how mean torques/forces vary with relative spike times DUE TO STIMULUS

#%% show variance in induced AP by superimposing 


#%% Wingbeat mean torques vs. stimulus time/phase 

plotchannels = ['mx','my','mz']

# Make aggregate control dictionary
aggdict = {}
for i in list(df.select_dtypes(include=np.number)): # loop over all numeric columns
    aggdict[i] = 'mean'
aggdict['wbstate'] = 'first'

# Keeping only stim wingbeats, take wingbeat means
dt = df.loc[df['wbstate']=='stim',].groupby(['wb','trial']).agg(aggdict)

# Remove DLM stimulation trials from 20210801
if date == '20210801':
    dt = dt.loc[~dt['trial'].isin([5,6,7,8]), ]

# Setup plot
fig, ax  = plt.subplots(3, 1, sharex='all', figsize=(4,8), gridspec_kw={'left' : 0.16})
viridis = cmx.get_cmap('viridis')

# Loop over moments, plot
for i,name in enumerate(plotchannels):
    ax[i].plot(dt['stimphase'], dt[name], '.')

# Plot aesthetics
for i,name in enumerate(plotchannels):
    ax[i].set_ylabel(name)
ax[0].set_xlim((0, 1))
ax[len(plotchannels)-1].set_xlabel('Stimulus phase')
# save
# plt.savefig(os.getcwd() + '/pics/' + date + '_stimphase_meanTorque.pdf',
#             dpi=500)




#%% Difference between traces 1wb pre, during, post stim



#%% A quickplot cell

quickPlot('20210803_1', '014',
          tstart=0, tend=20,
          plotnames=['stim','comparator','LDLM','RDLM','mx'])


#%% Mean traces over time, stimulus marked 
trial = 11

# Make aggregate control dictionary
aggdict = {}
for i in list(df.select_dtypes(include=np.number)): # loop over all numeric columns
    aggdict[i] = 'mean'
aggdict['wbstate'] = 'first'

bob = da.loc[da['trial']==trial,].groupby(['wb','trial']).agg(aggdict)


# Plot wb mean values for this trial
fig, ax = plt.subplots(len(channelsFT), 1, sharex='all')
for i, varname in enumerate(channelsFT):
    ax[i].plot(bob['Time'], bob[varname], marker='.')
# Replot stimulus wingbeats as red
for i, varname in enumerate(channelsFT):
    ax[i].plot(bob.loc[bob['wbstate']=='stim', 'Time'],
               bob.loc[bob['wbstate']=='stim', varname],
               'r.')
# Label axes
for i, varname in enumerate(channelsFT):
    ax[i].set_ylabel(varname)





'''
Bug fixes
- Long pauses between wingbeats get counted as single wingbeats. Need to remove those pauses
- Some traces (delay==4) grab more wingbeats than wbBefore requests (5 instead of 4)

TODO
- Optimize readin/processing; way too slow right now
- Move to non-pandas version? Make pandas dataframe only after processing?
- Change to allow arbitrary number of stimulus wingbeats (with some accidental skips)
- Create mean before-stim-after plots

- Mean vs stim phase: Change to also do DIFF from previous wingbeat
'''
