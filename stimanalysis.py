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
from scipy.signal import hilbert
from scipy.stats import binned_statistic
from pyfuncs import *

import time as systime



'''
Wingbeat removing conundrum.

Remove bad wingbeats early on in load/processing
Save a long bool vector (length of all data with nothing removed) of whether good or bad wingbeat

Run spike sort import the same, but them trim spikeBoolVec with that long vector

Note: iloc does not include last element in range, while loc does
'''

#%% Controls and Constants

#------ Global controls
# Plot controls
wbBefore = 4
wbAfter = 4
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
# names of wingbeat states that aren't just "regular"
states = ['pre', 'stim', 'post']
namesDLM = ['LDLM', 'RDLM']
namesDVM = ['LDVM', 'RDVM']

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
# wingbeats longer than this time in s are deemed pauses in flapping and removed
wblengththresh = 0.1


#------ Spike Sorting Controls
# Controls
plotRemoved = False

# Constants
stimwindow = 0.001  # s, spikes closer than this time to stim are removed
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

# DLM-DVM timing difference threshold
dtmaxthresh = 100  # (ms)


#%% Initial read and processing

# Loop over list of individuals
# rundates = ['20210714','20210721','20210727','20210730','20210801','20210803','20210803_1']
# rundates = ['20210816','20210816_1','20210817_1','20210818_1']
rundates = ['20210803_1','20210816','20210816_1','20210817_1','20210818_1']
for date in rundates:
    plt.close('all')

    print(date)
    print('Reading in main data...')
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
        # Put into global (for this trial) list
        goodwb.append(keep)
        
        
        # Remove bad wingbeats from dataframe
        
        # Remove bad wingbeats from wb2col
        
        
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
            da = da.append(dtemp, ignore_index=True)
            # Note: previously used pd.concat(), but somehow append is slightly faster

    #--- Calculate useful quantities/vectors
    # Wingbeat vectors
    wbinds = np.vstack(wbinds)
    goodwb = np.vstack(goodwb)
    wb = da['wb'].to_numpy()
    # length of each wingbeat (useful)
    wblen = da.groupby('wb')['wb'].transform('count').to_numpy()

    print('    done in ' + str(systime.perf_counter()-tic))
    
    # TODO: For handling multiple consecutive stims in DVM trials,
    # look at difference in stim times, assign those above threshold as different pulses
    

    #%% Pull in spike times from spike sorting
    print('Pulling and analyzing spike sorting...')
    tic = systime.perf_counter()

    # Load spike times for this date
    spikes, waveforms = readSpikeSort(date, readFrom=readFrom)

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
            firstrow = np.argmax(ida)

            # Turn spike times into indices
            spikeinds = np.rint(spikes[m][inds, 0]*fsamp).astype(int)
            # Save on boolean vector
            spikeBoolVec[spikeinds + firstrow] = True

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
                    spikeBoolVec[spikeinds[closestSpike] + firstrow] = False
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

    print('    done in ' + str(systime.perf_counter()-tic))
    
    
    #%% Remove 
    
    
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

    #%% Calculate DLM-DVM relative timing
    print('Relative DLM-DVM timing...')
    tic = systime.perf_counter()

    # plot controls
    cols = ['green', 'red', 'blue']

    # Setup, preallocation
    uniquewb = np.unique(da['wb'])
    wbstate = da['wbstate'].iloc[wbinds[:, 0]]
    firstDLM = np.zeros(2)
    firstDVM = np.zeros(2)
    dt = np.zeros((len(uniquewb), 2))  # L,R

    # Determine which muscles have spike sorted data
    hasSort = []
    for m in channelsEMG:
        hasSort.append(np.shape(spikes[m])[0] > 1)
    bothOnSide = [hasSort[0] & hasSort[1], hasSort[2] & hasSort[3]]
    # Loop over wingbeats
    for i, w in enumerate(uniquewb):
        # Grab this wingbeat's data
        thiswb = da.iloc[wbinds[i, 0]:wbinds[i, 1]] ################## assumes wingbeat number matches row index of wbinds

        # Loop over DLMs
        for j in np.where(bothOnSide)[0]:
            # Get first L&R DLM spike for each wingbeat
            firstDLM[j] = np.argmax(thiswb[namesDLM[j]+'_st'])
            # If there was no DLM spike, skip this wingbeat
            if firstDLM[j] == 0:
                dt[i, j] = np.nan
                continue
            # otherwise get where DVM spikes happen
            DVMspikes = np.where(thiswb[namesDVM[j]+'_st'])[0]
            # Continue if there WERE DVM spikes
            if len(DVMspikes) != 0:
                # first DVM spike occurring after that DLM spike
                firstDVM[j] = DVMspikes[np.argmax(DVMspikes > firstDLM[j])]
                # calculate deltas
                dt[i, j] = firstDVM[j] - firstDLM[j]
    # Change erroneous deltat's to nan
    dt[dt > dtmaxthresh/1000*fsamp] = np.nan
    dt[dt < 0] = np.nan

    # Assign deltas to column in da
    da['dtL'] = np.repeat(dt[:, 0], wblen[wbinds[:, 0]])
    da['dtR'] = np.repeat(dt[:, 1], wblen[wbinds[:, 0]])

    # plt.figure()
    # # Loop over states and plot each
    # for i, s in enumerate(states):
    #     instate = np.where(wbstate == s)[0]
    #     plt.hist(dt[instate, 0]/fsamp*1000, bins=100, color=cols[i], alpha=0.5)

    print('    done in ' + str(systime.perf_counter()-tic))

    #%% Plot timing difference against F/T variables
    tic = systime.perf_counter()
    # Make aggregation control dictionary
    aggdict = {}
    # Take first value of all variables
    for i in list(da):
        aggdict[i] = 'first'
    # except make FT variables take mean
    for i in channelsFT:
        aggdict[i] = 'mean'
    # aggregate dataframe
    df = da.loc[da['wbstate'].isin(states), ].groupby('wb').agg(aggdict)

    # Plot
    figL, axL = plt.subplots(len(channelsFT), len(states), figsize=(9, 9),
                              sharex=True, sharey='row')
    figR, axR = plt.subplots(len(channelsFT), len(states), figsize=(9, 9),
                              sharex=True, sharey='row')
    for j, s in enumerate(states):
        data = df.loc[df['wbstate'] == s, ]
        for i, m in enumerate(channelsFT):
            inds = data['dtL'] != 0
            # inds = np.logical_and(data['dtL']!=0, data['dtL'] < dtmaxthresh)
            axL[i, j].plot(data['dtL'][inds]/fsamp*1000,
                            data[m][inds], '.', markersize=0.8)
            # inds = np.logical_and(data['dtR']!=0, data['dtR'] < dtmaxthresh)
            inds = data['dtR'] != 0
            axR[i, j].plot(data['dtR'][inds]/fsamp*1000,
                            data[m][inds], '.', markersize=0.8)
    # Label plots
    for j, s in enumerate(states):
        axL[0, j].set_title(s)
        axR[0, j].set_title(s)
    axL[len(channelsFT)-1, 1].set_xlabel('DLM-DVM first spike time difference (ms)')
    axR[len(channelsFT)-1, 1].set_xlabel('DLM-DVM first spike time difference (ms)')
    for i, m in enumerate(channelsFT):
        axL[i, 0].set_ylabel(m)
        axR[i, 0].set_ylabel(m)

    print(systime.perf_counter()-tic)
    # Save plots
    if saveplots:
        plt.figure(figL.number)
        plt.savefig(savefigdir + 'dtL_vs_variables_'
                    + date + figFileType, dpi=dpi)
        plt.figure(figR.number)
        plt.savefig(savefigdir + 'dtR_vs_variables_'
                    + date + figFileType, dpi=dpi)

    #%% DLM-DVM spike timing difference, but with waveforms, colored by time difference

    plt.style.use('dark_background')
    # plot controls
    nbins = 5
    # Set up color scheme
    viridis = cmx.get_cmap('viridis')
    # which wingbeats have measured spike time diff
    hasdiff = np.all(~np.isnan(dt), axis=1)

    # Loop over F/T channels
    for plotvar in channelsFT:
        # Loop over left, right
        for ilr, lr in enumerate(['L', 'R']):
            # set up figures
            fig, ax = plt.subplots(1, len(states),
                                    sharex=True, sharey='row',
                                    figsize=(13, 7),
                                    gridspec_kw={'wspace': 0})

            # spike time difference range to set colors with
            # (simply grabbing dt's that aren't 0 or nan)
            dtgood = dt[((dt != 0) & ~np.isnan(dt))[:, ilr], ilr]
            if len(dtgood) != 0:
                stmin = np.min(dtgood)
                stmax = np.max(dtgood)
            else:
                continue
            # plot variable range to set spacing with
            plotvarScale = (np.max(da[plotvar]) - np.min(da[plotvar]))/2
            # Create bins
            bins = np.linspace(stmin, stmax, nbins)
            # Loop over states
            for j, s in enumerate(states):
                # Which wingbeats are in this state, have measured spike time diff
                thiswb = uniquewb[(hasdiff) & (wbstate == s)]
                # Loop over wingbeats
                for i, w in enumerate(thiswb):
                    # Determine spike time difference of this wingbeat
                    thisdt = da['dt'+lr].iloc[wbinds[w, 0]+1]
                    thisbin = np.digitize(thisdt, bins)
                    # Plot!
                    ax[j].plot(da['phase'].iloc[wbinds[w, 0]:wbinds[w, 1]],
                                da[plotvar].iloc[wbinds[w, 0]:wbinds[w, 1]]
                                + plotvarScale*thisbin,
                                lw=0.5, alpha=0.8,
                                color=viridis(thisbin/nbins))
            # Label timing difference bins
            for i in range(len(bins)):
                rangelow = f'{bins[i]/fsamp*1000:.1f}'
                if i == len(bins)-1:
                    rangehigh = '->'
                else:
                    rangehigh = f'{bins[i+1]/fsamp*1000:.1f}'
                ax[0].text(0.1, -plotvarScale/2 + plotvarScale*i, rangelow+'-'+rangehigh+' ms',
                            color='white')
            # Label overall plot
            ax[1].set_title(plotvar)

            if saveplots:
                plt.savefig(savefigdir+'staggeredWaveforms/'+plotvar+'_'+lr+'_'+date+figFileType,
                            dpi=dpi)

    # Set style back to normal
    plt.style.use('default')

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

    # TODO: for first spike in wingbeat?

    #%% Look at L-R timing differences for the same muscles

    # Overall histograms of L-R timing differences

    # L-R timing differences against mean output variables

    #%% Spike phase vs. stimphase

    # Spike phase stim only
    fig, ax = plt.subplots(len(channelsEMG), 1,
                            sharex=True, sharey=True,
                            figsize=(6, 9))

    for i, m in enumerate(channelsEMG):
        df = da.loc[(da['wbstate'] == 'stim')
                    & da[m+'_st'], ]
        ax[i].plot(df['stimphase'], df['phase'], '.', markersize=1)
        ax[i].set_ylabel(m)
    # save
    if saveplots:
        plt.savefig(savefigdir + 'stimphase_vs_spiketimes_' + date + figFileType,
                    dpi=dpi)

    #--- Spike phase pre-, stim, post-
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

    #%% show variance in induced AP by superimposing

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
    viridis = cmx.get_cmap('viridis')

    df = da.loc[(da['wbstate'].isin(states))
                & (da['pulse'] != 259), ]
    mincol = np.min(df['stimphase'])
    maxcol = np.max(df['stimphase'])

    plotvar = 'fz'

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
            # for j,m in enumerate(channelsFT):
            plt.plot(data['wb'], data['fz'] - data['fz'].iloc[0],
                      '-', marker='.',
                      alpha=0.4,
                      color=viridis(colphase))

    #%% Make plots of characterization trials


#%% A quickplot cell

# quickPlot('20210730', '014',
#           tstart=0, tend=20,
#           plotnames=['stim','LDVM','LDLM','RDLM','RDVM','mx'])

#%%

# wbmeanplot(channelsFT, da, 14)

'''
Bug fixes
- Long pauses between wingbeats get counted as single wingbeats. Need to remove those pauses
- Some traces (delay==4) grab more wingbeats than wbBefore requests (5 instead of 4)

TODO
- Optimize readin/processing; way too slow right now
- Move to non-pandas version? Make pandas dataframe only after processing?
- Change to allow arbitrary number of stimulus wingbeats (with some accidental skips)
- Handle spiek sorting _up and _down files without repeating spikes

- Mean vs stim phase: Change to also do DIFF from previous wingbeat
'''
