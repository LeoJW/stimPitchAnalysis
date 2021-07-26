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


# Manually entered delays
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
                    distance=wbdistance)[0]
    # Make long-form wingbeat column in dataframe (useful for stuff)
    dtemp['wb'] = 0
    dtemp.loc[wb, 'wb'] = 1
    dtemp['wb'] = np.cumsum(dtemp['wb'])
    
    # Make delay column
    dtemp['delay'] = delay[iabs]
    
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
    # Fit mean x wb pre-, during stim, and x wb after for all channels    
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
    dtemp = dtemp[dtemp['wbstate'].isin(['pre','stim','post'])]
    
    # Add to full dataframe
    if i==goodTrials[0]:
        df = dtemp
    else:
        df = pd.concat([df,dtemp])
        

#%%


binPlot(df,
        plotvars=channelsFT,
        groupvars=['wbstate','delay'],
        colorvar='delay',
        numbins=300, wbBefore=wbBefore, wbAfter=wbAfter,
        doSTD=False)

# # Make bin vectors
# nsamp = 300
# prebin = np.linspace(0, wbBefore, nsamp*wbBefore)
# stimbin = np.linspace(0, 1, nsamp)
# postbin = np.linspace(0, wbAfter, nsamp*wbAfter)


# # Color by delay controls
# colormax = np.max(delay)


# fig, ax = plt.subplots(len(channelsFT), 4,
#                        figsize=(15,10), squeeze=False,
#                        gridspec_kw={'width_ratios' : [wbBefore,1,wbAfter,0.01],
#                                     'wspace' : 0,
#                                     'left' : 0.05,
#                                     'right' : 1.0})
# viridis = cmx.get_cmap('viridis')

# # Loop over groups
# # for name, group in reversed(tuple(df.groupby(['wbstate', 'delay']))):
# for name, group in df.groupby(['wbstate', 'delay']):
#     # Loop over plotting variables
#     for i,varname in enumerate(channelsFT):
#         # Which axis to plot on, make binned means
#         # pre stim
#         if name[0]=='pre':
#             useax = 0
#             temp = group.groupby(np.digitize(group['multphase'], prebin)).agg(["mean","std"])
#         # stim
#         elif name[0]=='stim':
#             useax = 1        
#             temp = group.groupby(np.digitize(group['multphase'], stimbin)).agg(["mean","std"])
#         # post stim
#         else:
#             useax = 2
#             temp = group.groupby(np.digitize(group['multphase'], postbin)).agg(["mean","std"])
#         '''
#         NOTE:
#         The above code applies mean, std operation to EVERY column, including multphase
#         This means I'm plotting the MEAN of multphase per bin. Not wrong, but worth knowing
#         '''
        
            
        
#         # Plot STD shaded regions
#         ax[i,useax].fill_between(temp['multphase']['mean'],
#                                   temp[varname]['mean'] - temp[varname]['std'],
#                                   temp[varname]['mean'] + temp[varname]['std'],
#                                   color=viridis(name[1]/colormax)[0:3],
#                                   alpha=0.5)
#         # Plot mean lines
#         ax[i,useax].plot(temp['multphase']['mean'],
#                          temp[varname]['mean'],
#                          color=viridis(name[1]/colormax)[0:3],
#                          # alpha=0.7,
#                          lw=0.5)
#         # ax[i,useax].plot(group['multphase'],
#         #                  group[varname],
#         #                  color=viridis(name[1]/colormax)[0:3],
#         #                  alpha=0.3)


# for i,name in enumerate(channelsFT):
#     # Remove yaxis labels for rightmost 2 plots
#     ax[i,1].axes.get_yaxis().set_visible(False)
#     ax[i,2].axes.get_yaxis().set_visible(False)
#     # Set ylimits
#     yl = ax[i,0].get_ylim()
#     ax[i,1].set_ylim(yl)
#     ax[i,2].set_ylim(yl)
#     # Label y axes
#     ax[i,0].set_ylabel(name)
    
    
# # Colorbar
# tickrange = np.unique(np.sort(delay))
# cbar = fig.colorbar(cmx.ScalarMappable(norm=None, cmap=viridis),
#                     ax=ax[:],
#                     shrink=0.4,
#                     ticks=tickrange/colormax)
# cbar.ax.set_yticklabels(list(map(str, tickrange)),
#                         fontsize=7)
# cbar.ax.set_title('delay (ms)')

