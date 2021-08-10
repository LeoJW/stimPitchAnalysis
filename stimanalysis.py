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


#%%
# Loop over list of individuals
# rundates = ['20210714','20210721','20210727','20210730','20210801','20210803','20210803_1']
rundates = ['20210714']
for date in rundates:
    plt.close('all')
    
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
    zforceCutoffLPF = 40
    zforceCutoffHPF = 10
    wbdistance = 300 # samples
    fzrelheight = 0.25 # count peaks above this amount of min-max height
    
    # Thresholds
    stimthresh = 3 # threshold to count stim channel as "on"
    wblengththresh = 0.1 # wingbeats longer than this time in s are deemed pauses in flapping and removed
    
    # Known things
    states = ['pre','stim','post'] # names of wingbeat states that aren't just "regular"
    
    
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
    lastwb = 0
    lastwbind = 0
    wbinds = [] # global first,last index pairs for each wingbeat
    stiminds = [[] for x in range(goodTrials[-1]+1)] 
    ''' 
    ^Note the +1: For trials I'm sticking to a 1-indexing convention.
    This is evil in python, but it's easier to index like "trial 5 is data[trial==5]"
    given the way the trials were originally numbered!
    '''
    
    
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
        
        # Filter z force (manually apply twice for bandpass, builtin bandpass crashes)
        fzfilt = butterfilt(dtemp['fz'], zforceCutoffLPF, fsamp, order=4, bandtype='lowpass')
        fzfilt = butterfilt(fzfilt, zforceCutoffHPF, fsamp, order=4, bandtype='highpass')
        # Determine fz peak height for this trial
        maxfz = np.max(fzfilt)
        fzpeakheight = fzrelheight*maxfz
        # Grab wingbeats from filtered z force
        wb = find_peaks(fzfilt,
                        distance=wbdistance,
                        height=fzpeakheight)[0]
        
        # Save wingbeat indices
        wbinds.append(
            np.column_stack((
                np.insert(wb, 0, 0),
                np.insert(wb, len(wb), len(emg['Time'])-1)
                )) + lastwbind )
        # Update length of wingbeat inds
        lastwbind += len(emg['Time'])
        # Make long-form wingbeat column in dataframe (useful for stuff)
        dtemp['wb'] = 0
        dtemp.loc[wb, 'wb'] = 1
        dtemp['wb'] = np.cumsum(dtemp['wb']) + lastwb
        lastwb = dtemp['wb'].iloc[-1] + 1
        
        # # Test plot for diagnosing issues
        # plt.figure()
        # for j in np.unique(dtemp['wb']):
        #     if j % 2 == 0:
        #         wbtime = dtemp.loc[dtemp['wb']==j, 'Time'].to_numpy()
        #         plt.axvspan(wbtime[0], wbtime[-1], lw=0, color='#C2C2C2')
        # plt.plot(dtemp['Time'], fzfilt)            
        # plt.title(i)
        
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
        si = np.where(np.logical_and(dtemp['stim']>3,
                                     np.roll(dtemp['stim']<3, 1)))[0]
        stiminds[i] = si
    
        # Waste memory and create second phase column to count multiple wingbeats 
        # (0->1, 1->2, etc rather than 0->1, 0->1)
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
            stinds = dtemp['wb']==stimwb
            # Calculate stimulus delay and phase
            stdelay = (s - np.argmax(stinds))/fsamp*1000 #in ms
            stphase = stdelay*fsamp/1000/len(np.where(stinds)[0])
            # Change columns for stim wingbeat
            dtemp.loc[stinds, ['wbstate', 'pulse', 'stimdelay', 'stimphase']] = 'stim', pulsecount, stdelay, stphase
    
            
            #--- Pre-stim wingbeats
            # Grab indices of this pre-stim period
            inds = np.where(
                np.logical_and(
                    np.logical_and((dtemp['wb']>=(stimwb-wbBefore)), (dtemp['wb']<stimwb)),
                    (dtemp['wbstate']!='stim')))[0]
            # Change columns for pre-stim wingbeats
            dtemp.loc[inds, ['wbstate', 'pulse', 'stimdelay', 'stimphase']] = 'pre', pulsecount, stdelay, stphase
            
            
            #--- Post-stim wingbeats
            inds = np.where(
                np.logical_and(
                    np.logical_and((dtemp['wb']<=(stimwb+wbAfter)), (dtemp['wb']>stimwb)),
                    (dtemp['wbstate']!='stim')))[0]
            # Change columns for post-stim wingbeats
            dtemp.loc[inds, ['wbstate', 'pulse', 'stimdelay', 'stimphase']] = 'post', pulsecount, stdelay, stphase
            
            # Increment global pulse counter
            pulsecount += 1
            
            
        # Clean EMG channels to NAN during stimulation period (to remove artifact)
        for name in channelsEMG:
            dtemp.loc[dtemp['stim']>stimthresh, name] = np.nan
            
        # Remove wingbeats longer than a certain threshold
        # dtemp = dtemp[dtemp.groupby('wb')['reltime'].transform('count').lt(int(fsamp*wblengththresh))]
        
        # Add to full dataframe
        if i==goodTrials[0]:
            df = dtemp
        else:
            df = pd.concat([df,dtemp])
        print(systime.perf_counter()-tic)
            
    # Save to da (dataframe_all)
    da = df.copy()
    
            
    #--- Calculate useful quantities/vectors
    # Wingbeat vectors
    wbinds = np.vstack(wbinds)
    wb = da['wb'].to_numpy()
    # length of each wingbeat (useful)
    wblen = da.groupby('wb')['wb'].transform('count').to_numpy()

    
    # TODO: Look at difference in stim times, assign those above threshold as different pulses
    
    
    #%% Pull in spike times from spike sorting
    print('Pulling and analyzing spike sorting.....')
    tic = systime.perf_counter()
    
    # Controls
    plotRemoved = False
    
    # Constants
    stimwindow = 0.002 # s, spikes closer than this time to stim are removed
    skiptrials = {
        '20210730' : [22],
        '20210803_1' : [19,20]}
    '''
    QUICK FIX FOR 20210730: 22 is not in EMG, but in FT
    Make spikes from trial 22 empty, and shift the current 22-27 to 23-28
    
    LONG FIX: Redo spike sorting, alter offlineSortConvert to catch this
    
    Note that same issue should be present in 20210803_1
    '''
    
    # Load spike times for this date
    spikes, waveforms = readSpikeSort(date)
    
    
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
                spikes[m][spikes[m][:,1]>=i,1] += 1
        
        # Loop over trials that are also in goodtrials
        for i in list(set(np.unique(spikes[m][:,1])).intersection(goodTrials)):
            # Get inds that are in this trial for spikes and main dataframes
            inds = spikes[m][:,1]==i
            ida = da['trial']==i
            firstrow = np.argmax(ida)
            
            # Turn spike times into indices
            spikeinds = np.rint(spikes[m][inds,0]*fsamp).astype(int)
            # Save on boolean vector
            spikeBoolVec[spikeinds + firstrow] = True
            
            #--- Flip spike times to work on the -len : 0 time scale
            # Get time length of this trial
            tlength = np.min(da.loc[ida, 'Time'])
            # Flip times
            spikes[m][inds,0] = spikes[m][inds,0] + tlength
            
            #--- Remove spike times that fall within threshold of stim pulse
            stimtimes = da.loc[ida, 'Time'][stiminds[i]].to_numpy()
            closest = np.ones(len(stimtimes), dtype=int)
            for j in range(len(stimtimes)):
                spikeDistance = abs(spikes[m][inds,0] - stimtimes[j])
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
                plt.plot(waveforms[m][i,:])
            plt.title(m)
        
        # Actually remove spikes that are too close to stim
        spikes[m] = np.delete(spikes[m], (closespikes[m]), axis=0)
        waveforms[m] = np.delete(waveforms[m], (closespikes[m]), axis=0)
        
        # Update dataframe column with spikeBoolVec
        da[m+'_st'] = spikeBoolVec
        
    
    # Version without regular wingbeats 
    df = da[da['wbstate'].isin(states)]
    # Remove DLM stimulation trials from 20210801
    if date == '20210801':
        df = df.loc[~df['trial'].isin([5,6,7,8]), ]
        
    print('    done in ' + str(systime.perf_counter()-tic))
    
    #%% Calculate DLM-DVM relative timing
    print('Relative DLM-DVM timing')
    bigtic = systime.perf_counter()
    
    # plot controls
    cols = ['green','red','blue']
    
    # Known things
    namesDLM = ['LDLM', 'RDLM']
    namesDVM = ['LDVM', 'RDVM']
    
    # Setup, preallocation
    uniquewb = np.unique(da['wb'])
    wbstate = da['wbstate'].iloc[wbinds[:,0]]
    firstDLM = np.zeros(2)
    firstDVM = np.zeros(2)
    dt = np.zeros((len(uniquewb),2)) # L,R
    
    # Determine which muscles have spike sorted data
    hasSort = []
    for m in channelsEMG:
        hasSort.append(np.shape(spikes[m])[0] > 1)
    bothOnSide = [hasSort[0] & hasSort[1], hasSort[2] & hasSort[3]]
    # Loop over wingbeats
    for i,w in enumerate(uniquewb):        
        # Grab this wingbeat's data
        thiswb = da.iloc[wbinds[i,0]:wbinds[i,1]]
        
        # Loop over DLMs
        for j in np.where(bothOnSide)[0]:
            # Get first L&R DLM spike for each wingbeat
            firstDLM[j] = np.argmax(thiswb[namesDLM[j]+'_st'])
            # If there was no DLM spike, skip this wingbeat
            if firstDLM[j] == 0:
                dt[i,j] = np.nan
                continue
            # otherwise get where DVM spikes happen
            DVMspikes = np.where(thiswb[namesDVM[j]+'_st'])[0]
            # Continue if there WERE DVM spikes
            if len(DVMspikes) != 0:
                # first DVM spike occurring after that DLM spike
                firstDVM[j] = DVMspikes[np.argmax(DVMspikes > firstDLM[j])]
                # calculate deltas
                dt[i,j] = firstDVM[j] - firstDLM[j]
    
    # Assign deltas to column in da
    da['dtL'] = np.repeat(dt[:,0], wblen[wbinds[:,0]])
    da['dtR'] = np.repeat(dt[:,1], wblen[wbinds[:,0]])
    
    
    plt.figure()
    # Loop over states and plot each
    for i,s in enumerate(states):
        instate = np.where(wbstate==s)[0]
        plt.hist(dt[instate,0]/fsamp*1000, bins=100, color=cols[i], alpha=0.5)
    
    print('    done in ' + str(systime.perf_counter()-bigtic))
    
    
    #%% Plot timing difference against F/T variables
    
    # Make aggregation control dictionary
    aggdict = {}
    # Take first value of all variables
    for i in list(da):
        aggdict[i] = 'first'
    # except make FT variables take mean
    for i in channelsFT:
        aggdict[i] = 'mean'
    # aggregate dataframe
    dt = data.loc[data['trial']==trial,].groupby('wb').agg(aggdict)

    #%% Plot distribution of spike phase for each muscle 
    
    fig, ax = plt.subplots(len(channelsEMG), 1, sharex=True)
    for i,m in enumerate(channelsEMG):
        ax[i].hist(da.loc[da[m+'_st'], 'phase'], bins=100)
    # Labels and aesthetics
    ax[len(channelsEMG)-1].set_xlabel('Spike Phase')
    
    
    fig, ax = plt.subplots(len(channelsEMG), 1, sharex=True)
    for i,m in enumerate(channelsEMG):
        ax[i].hist(da.loc[da[m+'_st'] & 
                          (wblen<1000), 'reltime'], bins=100)
    # Labels
    ax[len(channelsEMG)-1].set_xlabel('Spike Time')
    
    
    # TODO: for first spike in wingbeat?
    
    
    
    #%% Spike phase vs. stimphase
    
    # Spike phase stim only
    fig, ax = plt.subplots(len(channelsEMG), 1,
                           sharex=True, sharey=True,
                           figsize=(6,9))
    
    for i,m in enumerate(channelsEMG):
        dt = da.loc[(da['wbstate']=='stim') & 
                    da[m+'_st'], ]
        ax[i].plot(dt['stimphase'], dt['phase'], '.')
        ax[i].set_ylabel(m)
    # save
    plt.savefig(os.path.dirname(__file__) + '/pics/' + 'stimphase_vs_spiketimes_' + date + '.pdf',
                dpi=500)
    
    
    #--- Spike phase pre-, stim, post-
    fig, ax = plt.subplots(len(channelsEMG), 3,
                           sharex=True, sharey=True,
                           figsize=(6,9),
                           gridspec_kw = {'wspace':0, 'hspace':0.1})
    # make plot
    for i,m in enumerate(channelsEMG):
        for j,s in enumerate(states):
            dt = da.loc[(da['wbstate']==s) & 
                        da[m+'_st'], ]
            ax[i,j].plot(dt['stimphase'], dt['phase'], '.')
    # labels, aesthetics
    for i,m in enumerate(channelsEMG):
        ax[i,0].set_ylabel(m)
    for j,s in enumerate(states):
        ax[len(channelsEMG)-1,j].set_xlabel(s)
    ax[0,0].set_xlim((0,1))
    ax[0,0].set_ylim((0,1))
    # save
    plt.savefig(os.path.dirname(__file__) + '/pics/' + 'stimphase_vs_spiketimes_prestimpost_' + date + '.pdf',
                dpi=500)
    
    
    
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
    dt = df.loc[df['wbstate']=='stim',].groupby('wb').agg(aggdict)
    
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
    plt.savefig(os.path.dirname(__file__) + '/pics/' + date + '_stimphase_meanTorque.pdf',
                dpi=500)
    
    
    
    
    #%% Difference between traces pre, during, post stim
    
    
    # set up figure
    plt.figure()
    viridis = cmx.get_cmap('viridis')
    
    dt = df.loc[df['pulse']!=259, ]
    mincol = np.min(dt['stimphase'])
    maxcol = np.max(dt['stimphase'])
    
    plotvar = 'fz'
    
    # Loop over pulses
    for i in np.unique(dt['pulse']):
        # Grab this pulse, take wb means
        data = dt.loc[dt['pulse']==i,].groupby(['wb']).agg(aggdict)
        data['wb'] = data['wb'] - data['wb'].iloc[0]
        # only continue if enough got picked up 
        if len(data) > wbBefore:
            # Color by phase
            colphase = (data['stimphase'].iloc[wbBefore] - mincol)/(maxcol - mincol)
            # Plot pre-stim-post sequence for this pulse
            # for j,m in enumerate(channelsFT):
            plt.plot(data['wb'], data['fz'] - data['fz'].iloc[wbBefore],
                       '-', marker='.',
                       alpha=0.4,
                       color=viridis(colphase))
        
    
#%% A quickplot cell

# quickPlot('20210730', '023',
#           tstart=0, tend=20,
#           plotnames=['stim','LDVM','LDLM','RDLM','RDVM','mx'])

    
    


#%%

dd = da.loc[(da['wbstate']=='pre') &
            (da['trial']==10), ]





'''
Bug fixes
- Long pauses between wingbeats get counted as single wingbeats. Need to remove those pauses
- Some traces (delay==4) grab more wingbeats than wbBefore requests (5 instead of 4)

TODO
- Optimize readin/processing; way too slow right now
- Move to non-pandas version? Make pandas dataframe only after processing?
- Change to allow arbitrary number of stimulus wingbeats (with some accidental skips)
- Handle spiek sorting _up and _down files without repeating spikes
- Create mean before-stim-after plots

- Mean vs stim phase: Change to also do DIFF from previous wingbeat
'''
