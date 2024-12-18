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
from scipy.signal import hilbert
from scipy.stats import binned_statistic
from sklearn.linear_model import LinearRegression
from pyfuncs import *

import time as systime



#%% Controls and Constants
lastwbnumber = 0
lastpulsenumber = 0


#------ Global controls
# Plot controls
wbBefore = 2
wbAfter = 2
# Figure saving controls
saveplots = True
savefigdir = os.path.dirname(__file__) + '/pics/'  # dir to save figures in
figFileType = '.png'
dpi = 300

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
zforceCutoff = [10,40]
wbdistance = 300  # samples
fzrelheight = 0.05  # count wingbeats where envelope > this amount of max
minwblen = 200 # minimum wingbeat length (no wbs faster than 50Hz)
maxwblen = 1000# maximum wingbeat length (no wbs slower than 10Hz)

# Thresholds
stimthresh = 3  # threshold to count stim channel as "on"


#------ Spike Sorting Controls
# Controls
plotRemoved = False

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
stimAmplitudeThresh = 7

# DLM-DVM timing difference threshold
dtmaxthresh = 100  # (ms)


# %% Initial read and processing

# Loop over list of individuals
# rundates = ['20210714','20210721','20210727','20210730','20210801','20210803','20210803_1']
# rundates = ['20210803_1','20210816','20210816_1','20210817_1','20210818_1']
# rundates = ['20210803_1','20210816','20210817_1','20210818_1']
rundates = ['20210803_1']
for date in rundates:
    plt.close('all')

    print(date)
    print('   Reading in main data...')
    tic = systime.perf_counter()

    #- Load data
    # Read empty FT for bias
    biasdata, colnames, fsamp = readMatFile(date, 'empty', doFT=True, readFrom=readFrom)
    bias = np.zeros((6, 1))
    for i in range(6):
        bias[i] = np.mean(biasdata[colnames[i+1]])
    # Read program guide to find good trials with delay
    goodTrials = whichTrials(date, readFrom=readFrom)

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
        # wbinds.append(wb2col + lastind)
        
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
        dtemp['stimdelay'] = 0
        dtemp['stimphase'] = 0
        # Label wingbeats as being pre-, during, or post- stimulation
        for s in si:
            # get which wingbeat this stim is on
            stimwb = dtemp.loc[s, 'wb']

            #--- Stim wingbeat
            # Grab indices
            stinds = dtemp['wb'] == stimwb
            # Calculate stimulus delay and phase
            stdelay = (s - np.argmax(stinds))/fsamp*1000  # in ms
            stphase = stdelay*fsamp/1000/len(np.where(stinds)[0])
            # Change columns for stim wingbeat
            dtemp.loc[stinds, ['wbstate', 'pulse', 'stimdelay',
                               'stimphase']] = 'stim', pulsecount, stdelay, stphase

            #--- Pre-stim wingbeats
            # Grab indices of this pre-stim period
            inds = np.where(
                np.logical_and(
                    np.logical_and(
                        (dtemp['wb'] >= (stimwb-wbBefore)), (dtemp['wb'] < stimwb)),
                    (dtemp['wbstate'] != 'stim')))[0]
            # Change columns for pre-stim wingbeats
            dtemp.loc[inds, ['wbstate', 'pulse', 'stimdelay',
                             'stimphase']] = 'pre', pulsecount, stdelay, stphase

            #--- Post-stim wingbeats
            inds = np.where(
                np.logical_and(
                    np.logical_and(
                        (dtemp['wb'] <= (stimwb+wbAfter)), (dtemp['wb'] > stimwb)),
                    (dtemp['wbstate'] != 'stim')))[0]
            # Change columns for post-stim wingbeats
            dtemp.loc[inds, ['wbstate', 'pulse', 'stimdelay',
                             'stimphase']] = 'post', pulsecount, stdelay, stphase

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

    #--- Calculate useful quantities/vectors
    # Wingbeat vectors
    wbinds = np.vstack(wbinds)
    goodwb = np.hstack(goodwb)

    print('       done in ' + str(systime.perf_counter()-tic))
    
    # TODO: For handling multiple consecutive stims in DVM trials,
    # look at difference in stim times, assign those above threshold as different pulses
    

#%% Pull in spike times from spike sorting
    print('   Pulling and analyzing spike sorting...')
    tic = systime.perf_counter()

    # Load spike times for this date
    spikes, waveforms = readSpikeSort(date,
                                      readFrom=readFrom,
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

        # Plot the spikes that are too close
        if plotRemoved:
            plt.figure()
            for i in closespikes[m]:
                plt.plot(waveforms[m][i, :])
            plt.title(m)

        # Remove spikes too close to stim from primary spike data objects as well
        spikes[m] = np.delete(spikes[m], (closespikes[m]), axis=0)
        waveforms[m] = np.delete(waveforms[m], (closespikes[m]), axis=0) 

        # Update dataframe column with spikeBoolVec
        da[m+'_st'] = spikeBoolVec

    # # Remove DLM stimulation trials from 20210801
    # if date == '20210801':
    #     df = df.loc[~df['trial'].isin([5,6,7,8]), ]

    print('       done in ' + str(systime.perf_counter()-tic))
    
#%% Keep only pulses where there is a good induced AP in both stim'd muscles
    windowLen = 50
    
    viridis = cmx.get_cmap('viridis')
    
    # Plot induced APs binned by stimphase
    nbin = 10
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
    if saveplots:
        fig.savefig(savefigdir + 'AP_comparison_' + date + figFileType,
                    dpi=dpi)
        figb.savefig(savefigdir + 'AP_comparison_reject_' + date + figFileType,
                      dpi=dpi)
                        
    #%% Remove bad wingbeats!
    da = da[goodwb]
    
    # Recreate a few vectors
    wblen = da.groupby('wb')['wb'].transform('count').to_numpy()
    wb = da['wb'].to_numpy()
    
    # #%% Save data to dropbox for Joy
    # print('Saving preprocessed data...')
    # tic = systime.perf_counter()
    
    # savedir = os.path.join(os.path.expanduser('~'),
    #                           'Dropbox (GaTech)',
    #                           'Sponberg Team',
    #                           'Leo Wood',
    #                           'pitchStim',
    #                           'dataPreprocessed')
    # # savedir = os.path.join('/Users/leo/Desktop/ResearchPhD/PitchControl/dataPreprocessed/')
    # # scipy.io.savemat(savedir + date + '.mat',
    # #                  {name: col.values for name, col in da.items()})
    # da.to_csv(savedir+date+'.csv')
    # print(systime.perf_counter()-tic)
    
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
    print('       done in ' + str(systime.perf_counter()-tic))


#%% Plot distribution of spike phase/time for each muscle

    # Spike phase
    fig, ax = plt.subplots(len(channelsEMG), 1, sharex=True)
    for i, m in enumerate(channelsEMG):
        ax[i].hist(da.loc[da[m+'_st'], 'phase'], bins=100, density=True)
        ax[i].set_ylabel(m)
    # Labels
    ax[len(channelsEMG)-1].set_xlabel('Spike Phase')

    # if saveplots:
    plt.savefig(savefigdir+'spikeDistributions/'+'phasehist_'+date+figFileType,
                dpi=dpi)

    # Spike time
    fig, ax = plt.subplots(len(channelsEMG), 1, sharex=True)
    for i, m in enumerate(channelsEMG):
        ax[i].hist(da.loc[da[m+'_st']
                          & (wblen < 1000), 'reltime'], bins=100)
    # Labels
    ax[len(channelsEMG)-1].set_xlabel('Spike Time')

    #%% Spike phase vs. stimphase

    # Spike phase pre-, stim, post-
    fig, ax = plt.subplots(len(channelsEMG), 3,
                            sharex=True, sharey=True,
                            figsize=(6, 9),
                            gridspec_kw={'wspace': 0, 'hspace': 0.1})
    # make plot
    for i, m in enumerate(channelsEMG):
        for j, s in enumerate(states):
            df = da.loc[(da['wbstate'] == s)
                        & da[m+'_st'], ]
            ax[i, j].plot(df['stimphase'], df['phase'], '.', markersize=1)
    # labels, aesthetics
    for i, m in enumerate(channelsEMG):
        ax[i, 0].set_ylabel(m)
    for j, s in enumerate(states):
        ax[len(channelsEMG)-1, j].set_xlabel(s)
    ax[0, 0].set_xlim((0, 1))
    ax[0, 0].set_ylim((0, 1))
    # save
    if saveplots:
        plt.savefig(savefigdir + 'stimphase_vs_spiketimes_prestimpost_' + date + figFileType,
                    dpi=dpi)
#%% First spike phase vs. stimphase
    
    # Spike phase pre-, stim, post-
    fig, ax = plt.subplots(len(hasSort), 3,
                            sharex=True, sharey=True,
                            figsize=(6, 9),
                            gridspec_kw={'wspace': 0, 'hspace': 0.1})
    # make plot
    for i, m in enumerate(hasSort):
        for j, s in enumerate(states):
            df = da.loc[(da['wbstate'] == s), ].groupby('wb').nth(0)
            ax[i, j].plot(df['stimphase'], df[m+'_fs']/10, '.', markersize=1)
    # labels, aesthetics
    for i, m in enumerate(hasSort):
        ax[i, 0].set_ylabel(m)
    for j, s in enumerate(states):
        ax[len(hasSort)-1, j].set_xlabel(s)
    # save
    if saveplots:
        plt.savefig(savefigdir + 'first_stimphase_spiketimes_' + date + figFileType,
                    dpi=dpi)
    
    # Overall histograms of L-R timing differences
    
    #%% Induced AP time lateral comparison
    aggdict = {}
    for i in list(da):
        aggdict[i] = 'first'
    df = da.loc[(da.wbstate=='stim') & (da.stimphase<0.5),].groupby('wb').aggregate(aggdict)
    
    plt.figure()
    plt.axline((0,0),(1,1), color='black')
    plt.plot(df.LDLM_fs/10, df.RDLM_fs/10, '.')
    plt.xlabel('LDLM time')
    plt.ylabel('RDLM time')
    plt.title(date)
    
    if saveplots:
        plt.savefig(savefigdir+'inducedAP_comparison_'+date+figFileType, dpi=dpi)

#%% L-R timing differences
    # Make aggregate control dictionary
    aggdict = {}
    for i in list(da):
        aggdict[i] = 'first'
    for i in channelsFT:  # loop over all numeric columns
        aggdict[i] = 'mean'
    # Create dataframe
    df = da.loc[(da.wbstate!='regular') & (da.stimphase<0.5),].groupby('wb').aggregate(aggdict)

    mincol = np.min(df['stimphase'])
    maxcol = np.max(df['stimphase'])
    
    # Create plots
    fig, ax = plt.subplots(2,1, figsize=(4.37,5.7))
    figDLM = plt.figure()
    axDLM = plt.gca()
    
    wbvec = []
    leftdata = []
    rightdata = []
    # Loop over pulses
    for i,p in enumerate(np.unique(df.pulse)):
        dt = df.loc[df.pulse==p,]
        # Shift wingbeats to line up at stim
        wbvec.append(dt['wb'].to_numpy() - dt.loc[dt['wbstate']=='stim','wb'].to_numpy())
        # Grab time differences
        leftdata.append((dt['LDVM_fs'].to_numpy()-dt['LDLM_fs'].to_numpy())/10)
        rightdata.append((dt['RDVM_fs'].to_numpy()-dt['RDLM_fs'].to_numpy())/10)
        # Color by stimulus phase
        colphase = (dt['stimphase'].iloc[wbBefore] - mincol)/(maxcol - mincol)
        ax[0].plot(wbvec[i], leftdata[i], '-', marker='.',
                   color=viridis(colphase), alpha=0.7)
        ax[1].plot(wbvec[i], rightdata[i], '-', marker='.',
                   color=viridis(colphase), alpha=0.7)
        
        axDLM.plot(wbvec[i], (dt['RDLM_fs'].to_numpy()-dt['LDLM_fs'].to_numpy())/10,
                   color=viridis(colphase), alpha=0.7)
    # Colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    tickrange = np.linspace(mincol, maxcol, 10)
    cbar = fig.colorbar(cmx.ScalarMappable(norm=None, cmap=viridis),
                        ticks=(tickrange-mincol)/(maxcol-mincol),
                        cax=cbar_ax)
    cbar.ax.set_yticklabels([str(round(x,3)) for x in tickrange], fontsize=7)
    cbar.ax.set_title('Phase of stimulus')
    # Labelling, aesthetics
    ax[0].set_ylabel('Left tDVM-tDLM')
    ax[1].set_ylabel('Right tDVM-tDLM')
    ax[0].set_ylim(bottom=0)
    ax[1].set_ylim(bottom=0)
    
    axDLM.set_ylabel('RDLM-LDLM')
    
    if saveplots:
        fig.savefig(savefigdir+'wb_vs_deltat_'+date+figFileType, dpi=dpi)
    
    
# #%% L-R timing differences against mean output variables
#     plotvar = 'mx'
#     # Summary stat to run
#     # summaryfunc = lambda x: np.nanmax(x) - np.nanmin(x)
#     summaryfunc = lambda x: np.trapz(x, dx=0.0001)
#     # Name to refer to it by
#     summaryStatName = 'integral'

#     # Make aggregate control dictionary
#     aggdict = {}
#     for i in list(da):
#         aggdict[i] = 'first'
#     for i in channelsFT:
#         aggdict[i] = summaryfunc
#     # Create dataframe
#     df = da.loc[(da.wbstate!='regular') & (da.stimphase<0.5),].groupby('wb').aggregate(aggdict)
#     mincol = np.min(df['stimphase'])
#     maxcol = np.max(df['stimphase'])
#     nwb = wbBefore+wbAfter+1
    
#     # Create plots
#     fig, ax = plt.subplots(2, wbBefore+wbAfter+1,
#                            sharex=True, sharey=True,
#                            figsize=(13,3),
#                            gridspec_kw={'wspace': 0, 'hspace': 0.1})
#     figd, axd = plt.subplots(2, wbBefore+wbAfter,
#                              sharex=True, sharey=True,
#                              figsize=(13,3),
#                              gridspec_kw={'wspace': 0, 'hspace': 0.1})
#     # Preallocate
#     npulse = len(np.unique(df.pulse))
#     tdl = np.full((npulse, wbBefore+wbAfter+1), np.nan)
#     tdr = np.full((npulse, wbBefore+wbAfter+1), np.nan)
#     val = np.full((npulse, wbBefore+wbAfter+1), np.nan)
#     difL = np.full((npulse, wbBefore+wbAfter), np.nan)
#     difR = np.full((npulse, wbBefore+wbAfter), np.nan)
#     difval = np.full((npulse, wbBefore+wbAfter), np.nan)
#     # Loop over pulses
#     for i,p in enumerate(np.unique(df.pulse)):
#         dt = df.loc[df.pulse==p,]
#         # grab wingbeats
#         thiswb = dt['wb'].to_numpy() - dt['wb'].iloc[0]
#         # Grab time differences
#         tdl[i,thiswb] = (dt['LDVM_fs'].to_numpy()-dt['LDLM_fs'].to_numpy())/10
#         tdr[i,thiswb] = (dt['RDVM_fs'].to_numpy()-dt['RDLM_fs'].to_numpy())/10
#         val[i,thiswb] = dt[plotvar].to_numpy()
#         # Color by stimulus phase
#         colphase = (dt['stimphase'].iloc[wbBefore] - mincol)/(maxcol - mincol)
#         # Loop over wingbeats, plot value for each wb
#         for j in range(nwb):
#             ax[0,j].plot(tdl[i,j], val[i,j], '.', color='black')
#             ax[1,j].plot(tdr[i,j], val[i,j], '.', color='black')
#         # Plot differences
#         difL[i,:] = np.diff(tdl[i,:])
#         difR[i,:] = np.diff(tdr[i,:])
#         difval[i,:] = np.diff(val[i,:])
#         for j in range(nwb-1):
#             axd[0,j].plot(difL[i,j], difval[i,j], '.', color='black')
#             axd[1,j].plot(difR[i,j], difval[i,j], '.', color='black')
        
#     # Labelling, aesthetics
#     ax[0,0].set_xlim(left=0)
#     fig.subplots_adjust(bottom=0.2)
#     ax[1,wbBefore].set_xlabel(r'$(t_{DVM}-t_{DLM})$')
#     for i in np.arange(-wbBefore,wbAfter+1):
#         ax[0,i+wbBefore].set_title(i)
#     ax[0,0].set_ylabel(summaryStatName+' '+plotvar+' Left')
#     ax[1,0].set_ylabel(summaryStatName+' '+plotvar+' Right')
    
#     for i in np.arange(-wbBefore,wbAfter):
#         axd[0,i+wbBefore].set_title(str(i)+r'$\rightarrow$'+str(i+1))
#     figd.subplots_adjust(bottom=0.2)
#     axd[1,wbBefore].set_xlabel(r'$\Delta (t_{DVM}-t_{DLM})$')
#     axd[0,0].set_ylabel(r'$\Delta$'+plotvar+' Left')
#     axd[1,0].set_ylabel(r'$\Delta$'+plotvar+' Right')
#     if date=='20210817_1':
#         axd[0,0].set_xlim(left=-10)
        
#     #--- summary statistics
#     coefs = np.full((wbBefore+wbAfter+1,4), np.nan) # wingbeat x Lslope, Lint, Rslope, Rint
#     dcoefs = np.full((wbBefore+wbAfter,4), np.nan)
#     cor = np.full((wbBefore+wbAfter+1,2), np.nan) # wingbeat x Lcor, Rcor
#     dcor = np.full((wbBefore+wbAfter,4), np.nan)
#     # Left side
#     if ~np.all(np.isnan(tdl)):
#         # Regular
#         for i in np.arange(0,wbBefore+wbAfter+1):
#             # Linear regression left
#             x = tdl[:,i]
#             y = val[:,i]
#             x,y = x[~np.isnan(x) & ~np.isnan(y)], y[~np.isnan(x) & ~np.isnan(y)]
#             reg = LinearRegression().fit(x.reshape(-1,1), y.reshape(-1,1))
#             coefs[i,0:2] = np.array([reg.coef_[0][0], reg.intercept_[0]])
#             # correlation left
#             cor[i,0] = np.corrcoef(x,y)[0,1]
#             # Plot linear regressions
#             ax[0,i].axline((0,coefs[i,1]), slope=coefs[i,0])
#             # Plot corr
#             ax[0,i].text(0.2, 0.7, 'r={:.2f}'.format(cor[i,0]), color='red', fontsize=8,
#                          horizontalalignment='center', verticalalignment='center',
#                          transform = ax[0,i].transAxes)
#         # Diff
#         for i in np.arange(0,wbBefore+wbAfter):
#             # Linear regression left
#             x = difL[:,i]
#             y = difval[:,i]
#             x,y = x[~np.isnan(x) & ~np.isnan(y)], y[~np.isnan(x) & ~np.isnan(y)]
#             reg = LinearRegression().fit(x.reshape(-1,1), y.reshape(-1,1))
#             dcoefs[i,0:2] = np.array([reg.coef_[0][0], reg.intercept_[0]])
#             # correlation left
#             dcor[i,0] = np.corrcoef(x,y)[0,1]
#             # Plot linear regressions
#             axd[0,i].axline((0,dcoefs[i,1]), slope=dcoefs[i,0])
#             # Plot corr
#             axd[0,i].text(0.2, 0.7, 'r={:.2f}'.format(dcor[i,0]), color='red', fontsize=8,
#                           horizontalalignment='center', verticalalignment='center',
#                           transform = axd[0,i].transAxes)
#     # Right side
#     if ~np.all(np.isnan(tdr)):
#         # Regular
#         for i in np.arange(0,wbBefore+wbAfter+1):
#             # Linear regression right
#             x = tdr[:,i]
#             y = val[:,i]
#             x,y = x[~np.isnan(x) & ~np.isnan(y)], y[~np.isnan(x) & ~np.isnan(y)]
#             reg = LinearRegression().fit(x.reshape(-1,1), y.reshape(-1,1))
#             coefs[i,2:4] = np.array([reg.coef_[0][0], reg.intercept_[0]])
#             # correlation right
#             cor[i,1] = np.corrcoef(x,y)[0,1]
#             # Plot linear regressions
#             ax[1,i].axline((0,coefs[i,3]), slope=coefs[i,2])
#             # Plot corr
#             ax[1,i].text(0.2, 0.7, 'r={:.2f}'.format(cor[i,1]), color='red', fontsize=8,
#                          horizontalalignment='center', verticalalignment='center',
#                          transform = ax[1,i].transAxes)
#         # Diff
#         for i in np.arange(0,wbBefore+wbAfter):
#             # Linear regression left
#             x = difR[:,i]
#             y = difval[:,i]
#             x,y = x[~np.isnan(x) & ~np.isnan(y)], y[~np.isnan(x) & ~np.isnan(y)]
#             reg = LinearRegression().fit(x.reshape(-1,1), y.reshape(-1,1))
#             dcoefs[i,2:4] = np.array([reg.coef_[0][0], reg.intercept_[0]])
#             # correlation left
#             dcor[i,1] = np.corrcoef(x,y)[0,1]
#             # Plot linear regressions
#             axd[1,i].axline((0,dcoefs[i,3]), slope=dcoefs[i,2])
#             # Plot corr
#             axd[1,i].text(0.2, 0.7, 'r={:.2f}'.format(dcor[i,1]), color='red', fontsize=8,
#                           horizontalalignment='center', verticalalignment='center',
#                           transform = axd[1,i].transAxes)
        
#     ''' 
#     NOTE:
#     could do everything above WAAY more elegantly. Worth a rewrite at a later date
#     '''
        
    
#     if saveplots:
#         fig.savefig(savefigdir+'dt_vs_'+summaryStatName+'_'+plotvar+'_'+date+figFileType,
#                     dpi=dpi)
#         figd.savefig(savefigdir+'delta_dt_vs_delta_'+summaryStatName+'_'+plotvar+'_'+date+figFileType,
#                      dpi=dpi)

#%% Waveforms colored/offset by DVM-DLM time differences
    plotvar = 'mx'
    # Create dataframe
    df = da.loc[(da.wbstate.isin(['stim','post'])) & (da.stimphase<0.5),].copy()
    df['tdl'] = (df['LDVM_fs']-df['LDLM_fs'])/10
    df['tdr'] = (df['RDVM_fs']-df['RDLM_fs'])/10
    df = df.loc[(df['tdl']>0) | (df['tdr']>0)]
    # mincol = np.nanmin((np.nanmin(df['tdl']), np.nanmin(df['tdr'])))
    # maxcol = np.nanmax((np.nanmax(df['tdl']), np.nanmax(df['tdr'])))
    mincol = 10
    maxcol = 40
    bins = np.linspace(0,1,10)
    
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True,
                           gridspec_kw={'wspace': 0, 'hspace': 0.1})
    
    # Loop over pulses
    for i,p in enumerate(np.unique(df.pulse)):
        dt = df.loc[df.pulse==p, ]
        # change phase to start at zero, go to wbBefore+wbAfter
        ph = dt.phase.to_numpy()
        for i in np.where(ph==1)[0]:
            ph[i+1:] += 1
        for j,side in enumerate(['l','r']):
            thiscolr = (dt['td'+side].iloc[0]-mincol)/(maxcol-mincol)
            if ~np.isnan(thiscolr):
                ax[j].plot(ph, dt.mx.to_numpy() + 4.5*np.digitize(thiscolr,bins),
                           color=viridis(thiscolr),
                           alpha=0.5, lw=0.5)
            
    # Colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    tickrange = np.linspace(mincol, maxcol, 10)
    cbar = fig.colorbar(cmx.ScalarMappable(norm=None, cmap=viridis),
                        ticks=(tickrange-mincol)/(maxcol-mincol),
                        cax=cbar_ax)
    cbar.ax.set_yticklabels([str(round(x,3)) for x in tickrange], fontsize=7)
    cbar.ax.set_title(r'$\Delta$t')
    
    if saveplots:
        fig.savefig(savefigdir+'waveforms_dt_'+date+figFileType, dpi=dpi)
    

#%% Wingbeat mean torques vs. stimulus time/phase

    plotchannels = ['mx', 'my', 'mz']

    # Make aggregate control dictionary
    aggdict = {}
    for i in list(da.select_dtypes(include=np.number)):  # loop over all numeric columns
        aggdict[i] = 'mean'
    aggdict['wbstate'] = 'first'

    # Keeping only stim wingbeats, take wingbeat means
    df = da.loc[da['wbstate'] == 'stim', ].groupby('wb').agg(aggdict)

    # Remove DLM stimulation trials from 20210801
    if date == '20210801':
        df = df.loc[~df['trial'].isin([5, 6, 7, 8]), ]

    # Setup plot
    fig, ax = plt.subplots(3, 1, sharex='all', figsize=(
        4, 8), gridspec_kw={'left': 0.16})
    viridis = cmx.get_cmap('viridis')

    # Loop over moments, plot
    for i, name in enumerate(plotchannels):
        ax[i].plot(df['stimphase'], df[name], '.')

    # Plot aesthetics
    for i, name in enumerate(plotchannels):
        ax[i].set_ylabel(name)
    ax[0].set_xlim((0, 1))
    ax[len(plotchannels)-1].set_xlabel('Stimulus phase')
    # save
    if saveplots:
        plt.savefig(savefigdir + date + '_stimphase_meanTorque' + figFileType,
                    dpi=dpi)


#%% Difference between traces pre, during, post stim
    
    # set up figure
    plt.figure()
    fig = plt.gcf()
    viridis = cmx.get_cmap('viridis')
    # Make aggregate control dictionary
    aggdict = {}
    for i in list(da.select_dtypes(include=np.number)):  # loop over all numeric columns
        aggdict[i] = 'std'
    aggdict['wbstate'] = 'first'
    aggdict['stimphase'] = 'first'
    aggdict['wb'] = 'first'
    aggdict['pulse'] = 'first'
    # Set up subsetted dataframe
    df = da.loc[da['wbstate'].isin(states) & (da['stimphase']<0.6), ]
    mincol = np.min(df['stimphase'])
    maxcol = np.max(df['stimphase'])
    
    plotvar = 'fz'
    zeroat = wbBefore
    
    # Loop over pulses
    for i in np.unique(df['pulse']):
        # Grab this pulse, take wb means
        data = df.loc[df['pulse'] == i, ].groupby(['wb']).agg(aggdict)
        data['wb'] = data['wb'] - data['wb'].iloc[0]
        # only continue if enough got picked up
        if len(data) > wbBefore:
            # Color by phase
            colphase = (data['stimphase'].iloc[wbBefore]
                        - mincol)/(maxcol - mincol)
            # Plot pre-stim-post sequence for this pulse
            plt.plot(data['wb'], data[plotvar] - data[plotvar].iloc[zeroat],
                     '-', marker='.',
                     alpha=0.4,
                     color=viridis(colphase))
    # Colorbar
    tickrange = np.linspace(mincol, maxcol, 10)
    cbar = fig.colorbar(cmx.ScalarMappable(norm=None, cmap=viridis),
                        ticks=(tickrange-mincol)/(maxcol-mincol))
    cbar.ax.set_yticklabels([str(round(x,3)) for x in tickrange], fontsize=7)
    cbar.ax.set_title('Phase of stimulus')
    
    # save
    if saveplots:
        plt.savefig(savefigdir+'diff_wbmean_'+aggdict['RDLM']+'_'+date+figFileType,
                    dpi=dpi)


    # Increment wingbeat, pulse, to make unique for each individual
    da['date'] = date
    da['wb'] += lastwbnumber
    da['pulse'] += lastpulsenumber
    lastwbnumber += np.max(da.wb)
    lastpulsenumber += np.max(da.pulse)
    # Aggregate into dataframe across all individuals
    if date==rundates[0]:
        df_all = da.loc[(da.wbstate!='regular') & (da.stimphase<0.5),]
    else:
        df_all = df_all.append(da.loc[(da.wbstate!='regular') & (da.stimphase<0.5),])

#%% Make plot with all individuals
plotvar = 'mx'
figFileType='.pdf'

totalDF = pd.DataFrame()
# Summary stat to run
sumFuncs = [lambda x: np.nanmax(x) - np.nanmin(x),
            lambda x: np.trapz(x, dx=0.0001),
            lambda x: np.mean(x),
            lambda x: np.std(x)]
sumNames = ['pkpk','integral','mean','std']
for ii in range(len(sumNames)):
    summaryfunc = sumFuncs[ii]
    summaryStatName = sumNames[ii]

    # Make aggregate control dictionary
    aggdict = {}
    for i in list(da):
        aggdict[i] = 'first'
    for i in channelsFT:
        aggdict[i] = summaryfunc
    # Create dataframe
    df = df_all.groupby(['wb','date']).aggregate(aggdict)
    mincol = np.min(df['stimphase'])
    maxcol = np.max(df['stimphase'])
    nwb = wbBefore+wbAfter+1
    
    # Create plots
    figd, axd = plt.subplots(2, wbBefore+wbAfter,
                            sharex=True, sharey=True,
                            figsize=(8,3),
                            gridspec_kw={'wspace': 0, 'hspace': 0.1})
    # Preallocate
    npulse = len(np.unique(df.pulse))
    tdl = np.full((npulse, wbBefore+wbAfter+1), np.nan)
    tdr = np.full((npulse, wbBefore+wbAfter+1), np.nan)
    val = np.full((npulse, wbBefore+wbAfter+1), np.nan)
    difL = np.full((npulse, wbBefore+wbAfter), np.nan)
    difR = np.full((npulse, wbBefore+wbAfter), np.nan)
    difval = np.full((npulse, wbBefore+wbAfter), np.nan)
    # Loop over pulses
    for i,p in enumerate(np.unique(df.pulse)):
        dt = df.loc[df.pulse==p,]
        # grab wingbeats
        thiswb = dt['wb'].to_numpy() - dt['wb'].iloc[0]
        # Grab time differences
        tdl[i,thiswb] = (dt['LDVM_fs'].to_numpy()-dt['LDLM_fs'].to_numpy())/10
        tdr[i,thiswb] = (dt['RDVM_fs'].to_numpy()-dt['RDLM_fs'].to_numpy())/10
        val[i,thiswb] = dt[plotvar].to_numpy()*1000
        # Color by stimulus phase
        colphase = (dt['stimphase'].iloc[wbBefore] - mincol)/(maxcol - mincol)
        # Plot differences
        difL[i,:] = np.diff(tdl[i,:])
        difR[i,:] = np.diff(tdr[i,:])
        difval[i,:] = np.diff(val[i,:])
        difL[i,difL[i,:]<-15] = np.nan
        difR[i,difR[i,:]<-15] = np.nan
        for j in range(nwb-1):
            axd[0,j].plot(difL[i,j], difval[i,j], '.', color='black')
            axd[1,j].plot(difR[i,j], difval[i,j], '.', color='black')
            
        theserows = pd.DataFrame({'dl':difL[i,:],
                                  'dr':difR[i,:],
                                  'val':difval[i,:],
                                  'wb':np.arange(0,4),
                                  'pulse':[p for x in range(4)],
                                  'date':[dt.date.iloc[0] for x in range(4)],
                                  'sum':[summaryStatName for x in range(4)]})
        totalDF = totalDF.append(theserows)
        
        
    # Labelling, aesthetics
    for i in np.arange(-wbBefore,wbAfter):
        axd[0,i+wbBefore].set_title(str(i)+r'$\rightarrow$'+str(i+1))
    figd.subplots_adjust(bottom=0.2)
    # axd[1,wbBefore].set_xlabel(r'$\Delta (t_{DVM}-t_{DLM})$')
    figd.text(0.5, 0.04, r'$\Delta (t_{DVM}-t_{DLM})$ (ms)', ha='center')
    figd.text(0.03, 0.5, r'$\Delta$ peak-peak $T_x$ (mNmm)', va='center', rotation='vertical')
    figd.text(0.06, 0.74, 'Left', va='center', rotation='vertical')
    figd.text(0.06, 0.39, 'Right', va='center', rotation='vertical')
    # axd[0,0].set_ylabel(r'Left')
    # axd[1,0].set_ylabel(r'Right')
    axd[0,0].set_xlim(left=-10)
        
    #--- summary statistics
    dcoefs = np.full((wbBefore+wbAfter,4), np.nan)
    dcor = np.full((wbBefore+wbAfter,4), np.nan)
    # Left side
    if ~np.all(np.isnan(tdl)):
        # Diff
        for i in np.arange(0,wbBefore+wbAfter):
            # Linear regression left
            x = difL[:,i]
            y = difval[:,i]
            x,y = x[~np.isnan(x) & ~np.isnan(y)], y[~np.isnan(x) & ~np.isnan(y)]
            reg = LinearRegression().fit(x.reshape(-1,1), y.reshape(-1,1))
            dcoefs[i,0:2] = np.array([reg.coef_[0][0], reg.intercept_[0]])
            # correlation left
            dcor[i,0] = np.corrcoef(x,y)[0,1]
            # Plot linear regressions
            axd[0,i].axline((0,dcoefs[i,1]), slope=dcoefs[i,0])
            # Plot corr
            axd[0,i].text(0.2, 0.7, 'r={:.2f}'.format(dcor[i,0]), color='red', fontsize=8,
                        horizontalalignment='center', verticalalignment='center',
                        transform = axd[0,i].transAxes)
    # Right side
    if ~np.all(np.isnan(tdr)):
        # Diff
        for i in np.arange(0,wbBefore+wbAfter):
            # Linear regression left
            x = difR[:,i]
            y = difval[:,i]
            x,y = x[~np.isnan(x) & ~np.isnan(y)], y[~np.isnan(x) & ~np.isnan(y)]
            reg = LinearRegression().fit(x.reshape(-1,1), y.reshape(-1,1))
            dcoefs[i,2:4] = np.array([reg.coef_[0][0], reg.intercept_[0]])
            # correlation left
            dcor[i,1] = np.corrcoef(x,y)[0,1]
            # Plot linear regressions
            axd[1,i].axline((0,dcoefs[i,3]), slope=dcoefs[i,2])
            # Plot corr
            axd[1,i].text(0.2, 0.7, 'r={:.2f}'.format(dcor[i,1]), color='red', fontsize=8,
                        horizontalalignment='center', verticalalignment='center',
                        transform = axd[1,i].transAxes)
            
    plt.savefig(savefigdir+'/joyThesisFigs/'+summaryStatName+figFileType, dpi=dpi)
    
totalDF.to_csv('/Users/leo/Desktop/ResearchPhD/PitchControl/analysis/forR.csv')

#%% A quickplot cell

trial = 10

quickPlot(date, str(trial).zfill(3),
          tstart=0, tend=20,
          lpfCutoff=500,
          plotnames=['stim','LDVM','LDLM','RDLM','RDVM','mx','fz','fy','fx'],
          readFrom='dropbox')

# set up
fig = plt.gcf()
df = da.loc[da.trial==trial, ]
ymin, ymax = np.nanmin(df['mx']), np.nanmax(df['mx'])
# Plot spike times
for i,m in enumerate(channelsEMG):
    plt.vlines(df.loc[df[m+'_st'], 'Time'],
              ymin=i+1, ymax=i+2, 
              color='black')
# Plot mean + sd of mx
for j in np.unique(df['wb']):
    wbtime = df.loc[df['wb']==j, 'Time'].to_numpy()
    
    yvec = df.loc[df['wb']==j, 'mx'].to_numpy()
    yvec = (yvec-ymin)/(ymax-ymin)
    plt.plot([wbtime[0], wbtime[-1]], np.mean(yvec)*np.ones(2)+5, color='black')
    plt.plot([wbtime[0], wbtime[-1]], np.mean(yvec)*np.ones(2)+5+np.std(yvec), color='gray')
    if j % 2 == 0:
        plt.axvspan(wbtime[0], wbtime[-1], lw=0, color='#C2C2C2')



'''
TODO
- Filter F/T BEFORE transform; would that matter?
- Change to allow arbitrary number of stimulus wingbeats (with some accidental skips)
- Handle spike sorting _up and _down files without repeating spikes
- Mean vs stim phase: Change to also do DIFF from previous wingbeat
'''

'''
SCATTERED NOTES
- spikes that are heavily drifted throughout cycle are likely cross-talk between DVM and DLM
Example: 20210816_1 RDVM, very clearly has cross-talk

'''
