#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 08 2021

Runs over selected dates, preprocesses data and saves a "cache" for later analysis

@author: leo
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import pandas as pd
from scipy.signal import hilbert
from pyfuncs import *

import time as systime


#------ Global controls
# Plot controls
wbBefore = 10
wbAfter = 10
# Figure saving controls
saveplots = False
figFileType = '.png'
dpi = 300
# Directory handling
filedir = os.path.dirname(__file__) # dir this file occupies
savefigdir = filedir + '/pics/'  # dir to save figures in

# Channel names to process
channelsEMG = ['LDVM', 'LDLM', 'RDLM', 'RDVM']
channelsExtra = ['stim']
channelsFT = ['fx', 'fy', 'fz', 'mx', 'my', 'mz']
# More known things
states = ['pre', 'stim', 'post'] # names of wingbeat states that aren't just "regular"

''' QUICK FIX FOR NOW '''
stimMuscles = ['LDLM','RDLM']

#------ Initial Read and Process Controls
readFrom = 'dropbox'
# Filter Controls
hpfCutoff = 70
lpfCutoff = 500
# Wingbeat finding
zforceCutoff = [10,40] # cutoff for filtering z force 
wbdistance = 300  # samples
fzrelheight = 0.05  # count wingbeats where envelope > this amount of max
minwblen = 200 # minimum wingbeat length (no wbs faster than 50Hz)
maxwblen = 1000# maximum wingbeat length (no wbs slower than 10Hz)
# Thresholds
stimthresh = 3  # threshold to count stim channel as "on"

#------ Spike Sorting Controls
# Constants
stimwindow = 0.0001  # s, spikes closer than this time to stim are removed
skiptrials = {
    '20210730': [22],
    '20210803_1': [19, 20],
    '20210816_1': [43]}
'''
QUICK FIX FOR 20210730: 22 is not in EMG, but in FT
Make spikes from trial 22 empty, and shift the current 22-27 to 23-28
LONG FIX: Redo spike sorting
Note that same issue is present in 20210803_1, 20210816_1
'''
stimAmplitudeThresh = 8 # spikes with amplitude above this are removed
dtmaxthresh = 100  # (ms) DLM-DVM timing difference threshold

#------ Filtering Data Controls
# Kept and rejected APs plot controls
nbin = 10 # How many stimphase bins to use
windowLen = 50 # Length of window after stim to plot 


runDates = ['20210803_1','20210816','20210817_1','20210818_1']
for date in runDates:
    #------------------------------------------------------------------------------------------------#
    '''
    READ DATA, INITIAL PROCESSING
    filtering, wingbeat finding, etc
    '''
    #------------------------------------------------------------------------------------------------#
    print(date)
    print('   Reading in main data...')
    tic = systime.perf_counter()

    # Load data
    # Read empty FT for bias
    biasdata, colnames, fsamp = readMatFile(date, 'empty', doFT=True, readFrom=readFrom)
    bias = np.zeros((6, 1))
    for i in range(6):
        bias[i] = np.mean(biasdata[colnames[i+1]])
    # Read program guide to find good trials with delay
    goodTrials = whichTrials(date, readFrom=readFrom)

    test = [[] for x in range(4)]
    # Create variables for loop
    pulsecount = 1
    lastwb = 0
    lastind = 0
    wbinds = []  # global first,last index pairs for each wingbeat
    stiminds = [[] for x in range(goodTrials[-1]+1)]
    goodwb = []
    '''
    ^Note the +1: For trials I'm sticking to a 1-indexing convention.
    This is evil in python, but it's easier to index like "trial 5 is data[trial==5]"
    given the way the trials were originally numbered!
    '''
    # Loop over all, read in data
    for i in goodTrials:
        # Make string version of trial
        trial = str(i).zfill(3)
        # Read data
        emg, emgnames, _ = readMatFile(date, trial, doFT=False, 
                                       grabOnly=channelsEMG+channelsExtra,
                                       readFrom=readFrom)
        ftd, ftdnames, _ = readMatFile(date, trial, doFT=True, bias=bias, readFrom=readFrom)
        # Filter data
        for name in channelsEMG:  # Filter EMG
            emg[name] = butterfilt(emg[name], hpfCutoff,
                                   fsamp, order=4, bandtype='high')
        for name in ftdnames:  # Filter FT
            ftd[name] = butterfilt(ftd[name], lpfCutoff,
                                   fsamp, order=4, bandtype='low')
        # Put everything together into a dataframe
        dtemp = pd.DataFrame({**emg, **ftd})
        
        # Filter z force
        fzfilt = cheby2filt(dtemp['fz'], zforceCutoff, fsamp, bandtype='bandpass')
        # Apply hilbert transform
        z = hilbert(fzfilt)
        # Define wingbeats as where phase angle crosses from negative->positive
        wb = np.where(np.diff(np.sign(np.angle(z)))==2)[0]
        
        # make two-column version of wb
        wb2col = np.column_stack((np.insert(wb, 0, 0),
                                  np.insert(wb, len(wb), len(emg['Time'])-1)))
        # Get length of wingbeats by taking diff
        wbdiff = np.diff(wb2col, axis=1).reshape(len(wb2col))
        # Make logical vector of where "bad" wingbeats are (too short, long, or fz below threshold)
        keep = np.full(len(dtemp), True)
        badwb = (np.abs(z)[wb2col[:,0]] < fzrelheight*np.max(fzfilt)) | \
                (wbdiff < minwblen) | \
                (wbdiff > maxwblen)
        for ii in np.where(badwb)[0]:
            keep[wb2col[ii,0]:wb2col[ii,1]] = False
            
        # Save into global (for this trial) list
        goodwb.append(keep)
        # Save wingbeat indices
        wbinds.append(wb2col[~badwb,:] + lastind)
        
        # Make long-form wingbeat column in dataframe 
        dtemp['wb'] = 0
        dtemp.loc[wb, 'wb'] = 1
        dtemp['wb'] = np.cumsum(dtemp['wb']) + lastwb
        lastwb = dtemp['wb'].iloc[-1] + 1
        # Make trial column
        dtemp['trial'] = i
        # Create phase column by rescaling time based on wingbeats
        dtemp['phase'] = np.nan
        for j in np.arange(len(wb)-1):
            ind = np.arange(wb[j], wb[j+1])
            dtemp.loc[ind, 'phase'] = np.linspace(0, 1, len(ind))
        # Create relative time column that starts at 0 every wingbeat
        dtemp['reltime'] = dtemp['Time']
        dtemp['reltime'] = dtemp.groupby(['wb'], group_keys=False).apply(
            lambda g: g['Time'] - g['Time'].iloc[0])

        # Get stim indices
        si = np.where(np.logical_and(dtemp['stim'] > 3,
                                     np.roll(dtemp['stim'] < 3, 1)))[0]
        stiminds[i] = si + lastind

        # Waste memory and create several different columns
        dtemp['wbstate'] = 'regular'
        dtemp['pulse'] = 0
        dtemp['stimphase'] = 0
        # Label wingbeats as being pre-, during, or post- stimulation
        for s in si:
            # get which wingbeat this stim is on
            stimwb = dtemp.loc[s, 'wb']

            #--- Stim wingbeat
            # Grab indices
            stinds = dtemp['wb'] == stimwb
            # Calculate stimulus delay and phase
            stphase = (s - np.argmax(stinds))/len(np.where(stinds)[0])
            # Change columns for stim wingbeat
            dtemp.loc[stinds, ['wbstate','pulse','stimphase']] = 'stim', pulsecount, stphase
            #--- Pre-stim wingbeats
            # Grab indices of this pre-stim period
            inds = np.where(
                np.logical_and(
                    np.logical_and(
                        (dtemp['wb'] >= (stimwb-wbBefore)), (dtemp['wb'] < stimwb)),
                    (dtemp['wbstate'] != 'stim')))[0]
            # Change columns for pre-stim wingbeats
            dtemp.loc[inds, ['wbstate','pulse','stimphase']] = 'pre', pulsecount, stphase
            #--- Post-stim wingbeats
            inds = np.where(
                np.logical_and(
                    np.logical_and(
                        (dtemp['wb'] <= (stimwb+wbAfter)), (dtemp['wb'] > stimwb)),
                    (dtemp['wbstate'] != 'stim')))[0]
            # Change columns for post-stim wingbeats
            dtemp.loc[inds, ['wbstate','pulse','stimphase']] = 'post', pulsecount, stphase
            # Increment global pulse counter
            pulsecount += 1

        # Clean EMG channels to NAN during stimulation period (to remove artifact)
        for name in channelsEMG:
            dtemp.loc[dtemp['stim'] > stimthresh, name] = np.nan
        
        # Increment indices to match when this trial is added to all trials
        dtemp.index = pd.RangeIndex(start=lastind, stop=lastind+len(emg['Time']), step=1)
        # Update length of this trial 
        lastind += len(emg['Time'])

        # Add to full dataframe
        if i == goodTrials[0]:
            da = dtemp
        else:
            da = da.append(dtemp)
            # Note: previously used pd.concat(), but somehow append is slightly faster

    # Wingbeat vectors (useful as standalone for speeding up other things)
    wbinds = np.vstack(wbinds)
    goodwb = np.hstack(goodwb)

    print('       done in ' + str(systime.perf_counter()-tic))
    
    #------------------------------------------------------------------------------------------------#
    '''
    Pull in spike times from spike sorting
    '''
    #------------------------------------------------------------------------------------------------#

    print('   Pulling and analyzing spike sorting...')
    tic = systime.perf_counter()

    # Load spike times for this date
    spikes, waveforms = readSpikeSort(date,readFrom=readFrom,
                                    stimAmplitudeThresh=stimAmplitudeThresh)

    # Preparation, preallocation
    closespikes = {}
    susrows = {}
    for m in channelsEMG:
        # dicts to grab spikes too close to stim
        closespikes[m] = []
        susrows[m] = []
        # New columns in dataframes for spike times (st's)
        da[m+'_st'] = False

    # Loop over muscles
    for m in channelsEMG:
        # Make boolean array that'll be true wherever spikes are
        spikeBoolVec = np.zeros(len(da), dtype=bool)

        # Fix skiptrials issues if present
        if date in list(skiptrials.keys()):
            # Loop over each skip
            for i in skiptrials[date]:
                # Bump all trials up one to skip this trial
                spikes[m][spikes[m][:, 1] >= i, 1] += 1

        # Loop over trials that are also in goodtrials
        for i in list(set(np.unique(spikes[m][:, 1])).intersection(goodTrials)):
            # Get inds that are in this trial for spikes and main dataframe
            inds = spikes[m][:, 1] == i
            ida = da['trial'] == i
            firstrow = da.index[np.argmax(ida)]

            # Turn spike times into indices
            spikeinds = np.rint(spikes[m][inds, 0]*fsamp).astype(int) + firstrow
            # Save on boolean vector
            spikeBoolVec[spikeinds] = True

            #--- Flip spike times to work on the -len : 0 time scale
            # Get time length of this trial
            tlength = np.min(da.loc[ida, 'Time'])
            # Flip times
            spikes[m][inds, 0] = spikes[m][inds, 0] + tlength

            #--- Remove spike times that fall within threshold of stim pulse
            stimtimes = da.loc[ida, 'Time'][stiminds[i]].to_numpy()
            closest = np.ones(len(stimtimes), dtype=int)
            for j in range(len(stimtimes)):
                spikeDistance = abs(spikes[m][inds, 0] - stimtimes[j])
                closestSpike = np.argmin(spikeDistance)
                # If none actually met condition, change value to reflect that
                if spikeDistance[closestSpike] > stimwindow:
                    closest[j] = -1
                else:
                    # Otherwise save spike that was closest
                    closest[j] = np.where(inds)[0][closestSpike]
                    # Remove from boolean vector
                    spikeBoolVec[spikeinds[closestSpike]] = False
            # Save closest spikes
            closespikes[m].extend(closest[closest != -1].tolist())

        # Remove spikes too close to stim from primary spike data objects as well
        spikes[m] = np.delete(spikes[m], (closespikes[m]), axis=0)
        waveforms[m] = np.delete(waveforms[m], (closespikes[m]), axis=0) 

        # Update dataframe column with spikeBoolVec
        da[m+'_st'] = spikeBoolVec

    # # Remove DLM stimulation trials from 20210801
    # if date == '20210801':
    #     df = df.loc[~df['trial'].isin([5,6,7,8]), ]

    print('       done in ' + str(systime.perf_counter()-tic))

    #------------------------------------------------------------------------------------------------#
    '''
    FILTER DATA AND SAVE
    Keep only pulses where there are good induced APs in both stimulated muscles
    '''
    #------------------------------------------------------------------------------------------------#
    if saveplots:
        # Plot induced APs binned by stimphase
        viridis = cmx.get_cmap('viridis')
        bins = np.linspace(0, 1, nbin)
        fig, ax = plt.subplots(len(stimMuscles), nbin,
                                figsize=(14.4,3.6),
                                gridspec_kw={'wspace': 0, 'hspace': 0.1})
        figb, axb = plt.subplots(len(stimMuscles), nbin,
                                figsize=(14.4,3.6),
                                gridspec_kw={'wspace': 0, 'hspace': 0.1})
        # Loop over trials
        for i in range(len(stiminds)):
            # Continue only if there were stims in this trial
            if len(stiminds[i])>0:
                # Loop over stim'd muscles
                for im,m in enumerate(stimMuscles):
                    # Loop over stim inds in this trial
                    for j in stiminds[i]:
                        # Get where first spike happens within window
                        firstSpike = np.argmax(da.loc[j:j+windowLen, m+'_st'])
                        # If no spike within window, count this pulse to be removed
                        if firstSpike==0:
                            # Set this pulse to be counted as "bad" and filtered out
                            goodwb[da.pulse==da.loc[j,'pulse']] = False
                            # Add to plot
                            thisbin = np.digitize(da.loc[j,'stimphase'], bins) - 1
                            axb[im,thisbin].plot(np.arange(0,100+1), da.loc[j:j+100,m].to_numpy(),
                                                lw=0.5, alpha=0.5,
                                                color=viridis(da.loc[j,'stimphase']))
                        else:
                            # Add to plot
                            thisbin = np.digitize(da.loc[j,'stimphase'], bins) - 1
                            ax[im,thisbin].plot(np.arange(0,100+1), da.loc[j:j+100,m].to_numpy(),
                                                lw=0.5, alpha=0.5,
                                                color=viridis(da.loc[j,'stimphase']))
        # labelling
        for im,m in enumerate(stimMuscles):
            ax[im,0].set_ylabel(m)
            axb[im,0].set_ylabel(m+' bad')
            for i in range(nbin-1):
                ax[im,i+1].get_yaxis().set_ticks([])
                axb[im,i+1].get_yaxis().set_ticks([])
        # save
        fig.savefig(savefigdir + 'AP_comparison_' + date + figFileType,
                    dpi=dpi)
        figb.savefig(savefigdir + 'AP_comparison_reject_' + date + figFileType,
                    dpi=dpi)
                        
    # Remove bad wingbeats!
    da = da[goodwb]
    
    # Recreate a few vectors
    wblen = da.groupby('wb')['wb'].transform('count').to_numpy()
    wb = da['wb'].to_numpy()
    
    #%% Grab first spike per wingbeat
    difthresh = 30 # 3ms
    print('   Grabbing first spike per wingbeat...')
    tic = systime.perf_counter()
    
    # Determine which muscles were spike sorted
    hasSort = [m for m in channelsEMG if np.shape(spikes[m])[0] > 1]
    # preallocate first_spike columns
    for m in channelsEMG:
        da[m+'_fs'] = np.nan
    # Loop over muscles
    for i,m in enumerate(hasSort):
        firstall = []
        # Loop over trials
        for j in np.unique(da.trial):
            # get indices 
            dt = da.loc[da.trial==j,]
            # get index of first spike in all wingbeats
            first = dt.groupby('wb')[m+'_st'].idxmax() - \
                dt.groupby('wb')[m+'_st'].idxmin()
            # Note which wingbeats to ignore based on state (stim, prestim, etc)
            ignorewb = dt.groupby('wb')['wbstate'].nth(0)!='regular'
            # note which wingbeats have diff outside threshold
            difBad = np.insert(np.diff(first) > difthresh, 0, True)
            # Remove zeros, those with diff outside threshold (that aren't stim!)
            first[(first==0) | difBad | ~ignorewb] = np.nan
            firstall.append(first.to_numpy())
        firstall = np.hstack(firstall)
        da[m+'_fs'] = np.repeat(firstall, wblen[np.insert(np.diff(wb)!=0,0,True)])

    # Remove wingbeats that aren't near stimulus
    da = da.loc[da.wbstate!='regular',]

    #--- Save
    # If cache dir hasn't been made, make it
    if 'preprocessedCache' not in os.listdir(filedir):
        os.mkdir(filedir + '/preprocessedCache')
    # Save to cache dir
    da.to_hdf(os.path.join(filedir, 'preprocessedCache', date)+'.h5',
            key='df', mode='w')


    print('       done in ' + str(systime.perf_counter()-tic))



'''
Problems to solve:

- How does the program know which muscles are stim'd on which trials?
how does this deal with the occasional multi-stim?

- 
'''