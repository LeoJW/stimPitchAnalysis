#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 11:13:29 2021

@author: leo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.gridspec as gridspec

import os
import scipy.io
import csv
from scipy.signal import butter, filtfilt, cheby2



# Grab data from a trial. Assumes data dirs are one dir back from location of this script
def readMatFile(date, trial, doFT=False, bias=np.zeros((6,1)),
                grabOnly=[]):
    # Move to data directory
    startdir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(os.path.join(startdir, os.pardir + '/' + date))
    # Recording type
    if doFT:
        recname = 'FT'
    else:
        recname = 'recording'
    # Handle underscores for dates with more than 1 moth
    if '_' in date:
        date = date[:len(date)-2]
    # Set up filename
    fname = 'Moth_EMG_' + recname + '_' + date + '_' + trial
    # Load mat file
    mat = scipy.io.loadmat(fname + '.mat')
    # Grab filename from first dict key that doesn't start with '_'
    fname = next((k for k in mat.keys() if k[0]!='_'), None)    
    # Grab data
    datamat = mat[fname]
    # Grab names 
    # (don't ask about the zeros, MANY wrapping array layers)
    names = []
    for column in mat[fname+'_Header'][0][0][0][0]:
        names.append(column[0])
    
    # If FT data, then calibrate/transform, change names
    if doFT:
        datamat[:,1:-1] = transformFTdata(datamat[:,1:-1].transpose(), bias)
        names[1:-1] = ['fx','fy','fz','mx','my','mz']
    
    # If no specific channels requested, return all
    if len(grabOnly)==0:
        # Put both together into dictionary
        d = {}
        for i in range(len(names)):
            d[names[i]] = datamat[:,i]
    # otherwise grab only requested main channels
    else:
        d = {}
        # Find indices matching requested strings
        inds = [i for i,x in enumerate(names) if any(xs in x for xs in grabOnly)]
        # Append time channel
        inds.insert(0, 0)
        for i in inds:
            d[names[i]] = datamat[:,i]
    # Grab sample rate
    fsamp = mat[fname+'_Control'][0][0][3][0][0]
    # Move back to starting directory (assuming same as script calling this function!)
    os.chdir(startdir)
    # Return
    return d, names, fsamp



# Grab spike-sorted data, consisting of spike times and waveforms
def readSpikeSort(date, muscles=['LDVM','LDLM','RDLM','RDVM']):
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
                rminds = np.where(np.any(mat[ch][:,2:] > 9, axis=1))[0]
                np.delete(temparray, (rminds), axis=0)
                np.delete(mat[ch], (rminds), axis=0)
                
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
    

# grab which trials for this moth are good and have delay
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
            end = table[3][0]
        # Create range
        trials = np.arange(start, end+1)
        # Remove any characterization that may have happened in middle
        if len(table[1])>2:
            # Loop over how many periods may have happened
            for i in np.arange(2, len(table[1]), 2):                
                trials = [x for x in trials if x<table[1][i] or x>table[1][i+1]]
        # Remove any error trials, if they exist
        if 'error' in names:
            # Remove trials marked as error
            for i in table[4]:
                trials = [x for x in trials if x!=i]
                
        return trials
    # TODO extend to return other types of trials (stim characterization) eventually
    
    
    


'''
TRANSFORMFTDATA:
Convert raw voltages of force/torque transducer  to real force and torque values
Uses ATI calibration matrix, bias offset settings, and translation matrix of axes

Input variables:
rawData = 6 x N np matrix of gauge voltages at single time point
biasOffset = 6 x 1 np matrix of offset correction voltage values

Output Variables:
values = 6 x N matrix of transformed force/torque values in N and Nmm
'''
def transformFTdata(rawData, biasOffset):
    # Calibration Matrix, in N and N-mm:
    cal_m = np.array([
        [-0.000352378, 0.020472451, -0.02633045, -0.688977299, 0.000378075, 0.710008955],
        [-0.019191418, 0.839003543, -0.017177775, -0.37643613, 0.004482987, -0.434163392],
        [ 0.830046806, 0.004569748, 0.833562339, 0.021075403, 0.802936538, -0.001350335],
        [-0.316303442, 5.061378026, 4.614179159, -2.150699522, -4.341889297, -2.630773662],
        [-5.320003676, -0.156640061, 2.796170871, 4.206523866, 2.780562472, -4.252850011],
        [-0.056240509, 3.091367987, 0.122101875, 2.941467741, 0.005876647, 3.094672928]
        ])
    trans_m = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0,19, 0, 1, 0, 0],
        [-19,0,0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
        ])
    
    # Calibrate
    rawData = np.matmul(cal_m, rawData)
    # Translate
    rawData = np.matmul(trans_m, rawData)
    # Apply bias offset
    rawData = rawData - biasOffset
    # Finish
    return(rawData.transpose())



'''
Filtering convenience functions
'''
def butterfilt(signal, cutoff, fs, order=4, bandtype='low'):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype=bandtype, analog=False)
    y = filtfilt(b, a, signal)
    return y
def cheby2filt(signal, cutoff, fs, rs=40, order=4, bandtype='low'):
    nyq = 0.5 * fs
    b, a = cheby2(order, rs, cutoff/nyq, btype=bandtype, analog=False)
    y = filtfilt(b, a, signal)
    return y




'''
Plotting utility for lines with character bars on end. Taken from
https://stackoverflow.com/questions/52743119/line-end-styles-in-matplotlib
'''
import matplotlib as mpl
import matplotlib.pyplot as plt

def add_interval(ax, xdata, ydata, caps="  ",
                 color='black', capsize=24):
    line = ax.add_line(mpl.lines.Line2D(xdata, ydata, color=color, solid_capstyle='butt'))
    anno_args = {
        'ha': 'center',
        'va': 'center',
        'size': capsize,
        'color': line.get_color()
    }
    a0 = ax.annotate(caps[0], xy=(xdata[0], ydata[0]), **anno_args)
    a1 = ax.annotate(caps[1], xy=(xdata[1], ydata[1]), **anno_args)
    return (line,(a0,a1))

    

def quickPlot(date, trial, tstart=5, tend=10,
              plotnames=['LDVM', 'LDLM','RDLM','RDVM'],
              hpfCutoff = 70,
              lpfCutoff = 500):
    # Dumb hard-coded info
    channelsEMG = ['LDVM','LDLM','RDLM','RDVM']
    # Read empty data for FT
    biasdata, colnames, fsamp = readMatFile(date, 'empty', doFT=True)
    bias = np.zeros((6,1))
    for i in range(6):
        bias[i] = np.mean(biasdata[colnames[i+1]])
    # Read actual data
    emg, emgnames, fsamp = readMatFile(date, trial, doFT=False)
    ftd, ftdnames, _     = readMatFile(date, trial, doFT=True)
    # Filter data
    for name in channelsEMG: # Filter EMG
        emg[name] = butterfilt(emg[name], hpfCutoff, fsamp, order=4, bandtype='high')
    for name in ftdnames: # Filter FT
        ftd[name] = butterfilt(ftd[name], lpfCutoff, fsamp, order=4, bandtype='low')
    # Remove stim periods from emg channels
    inds = emg['stim'] > 3
    for name in channelsEMG:
        emg[name][inds] = np.nan
    # Combine into single dict
    full = {**emg, **ftd}
    # Find indices to plot
    inds = np.arange(tstart*fsamp, tend*fsamp)
    # Plot
    plt.figure()
    for i,n in enumerate(plotnames):
        # Rescale y to 0-1
        # TODO: Rescaling not working! need to remove stim artifact
        yvec = full[n][inds]
        yvec = (yvec-np.nanmin(yvec))/(np.nanmax(yvec)-np.nanmin(yvec))
        plt.plot(full['Time'][inds], yvec+i)
    # aesthetics
    plt.title(date + '-' + trial)
    plt.show()

        
# binPlot: More general-purpose plotting of dataframes
def binPlot(df,
            plotvars, groupvars, colorvar,
            numbins, wbBefore, wbAfter,
            doSTD=True,
            doSummaryStat=True):
    # Make bins
    prebin = np.linspace(0, wbBefore, numbins*wbBefore)
    stimbin = np.linspace(0, 1, numbins)
    postbin = np.linspace(0, wbAfter, numbins*wbAfter)

    # Color by delay controls
    colormax = np.max(df[colorvar])
    colormin = np.min(df[colorvar])
    # Make plot
    fig, ax = plt.subplots(len(plotvars), 4,
                           figsize=(15,10), squeeze=False,
                           gridspec_kw={'width_ratios' : [wbBefore,1,wbAfter,0.01],
                                        'wspace' : 0})
    viridis = cmx.get_cmap('viridis')
    
    # Version that takes summary stats:
    if doSummaryStat:    
        # Loop over groups
        for name, group in df.groupby(groupvars):
            # Loop over plotting variables
            for i,varname in enumerate(plotvars):
                    # Which axis to plot on, make binned means
                    # pre stim
                    if name[0]=='pre':
                        useax = 0
                        temp = group.groupby(np.digitize(group['multphase'], prebin)).agg(["mean","std"])
                    # stim
                    elif name[0]=='stim':
                        useax = 1        
                        temp = group.groupby(np.digitize(group['multphase'], stimbin)).agg(["mean","std"])
                    # post stim
                    else:
                        useax = 2
                        temp = group.groupby(np.digitize(group['multphase'], postbin)).agg(["mean","std"])
                    '''
                    NOTE:
                    The above code applies mean, std operation to EVERY column, including multphase
                    This means I'm plotting the MEAN of multphase per bin. Not wrong, but worth knowing
                    '''
                    
                    # Plot STD shaded regions
                    if doSTD:
                        ax[i,useax].fill_between(temp['multphase']['mean'],
                                                  temp[varname]['mean'] - temp[varname]['std'],
                                                  temp[varname]['mean'] + temp[varname]['std'],
                                                  color=viridis(name[1]/colormax)[0:3],
                                                  alpha=0.5)
                    # Plot mean lines
                    ax[i,useax].plot(temp['multphase']['mean'],
                                     temp[varname]['mean'],
                                     color=viridis((name[1]-colormin)/colormax)[0:3],
                                     lw=0.5)
    # Version without summary stats
    else:
        # Loop over groups
        for name, group in df.groupby(groupvars):
            # Loop over plotting variables
            for i,varname in enumerate(plotvars):
                # Choose axis 
                if name[0]=='pre':
                    useax = 0
                elif name[0]=='stim':
                    useax = 1     
                else:
                    useax = 2
                # Plot!
                ax[i,useax].plot(group['multphase'],
                                 group[varname],
                                 color=viridis((group[colorvar].iloc[0]-colormin)/colormax)[0:3],
                                 lw=0.5)
                    
    # Axis handling    
    for i,name in enumerate(plotvars):
        # Remove yaxis labels for rightmost 2 plots
        ax[i,1].axes.get_yaxis().set_visible(False)
        ax[i,2].axes.get_yaxis().set_visible(False)
        # Set ylimits
        yl = ax[i,0].get_ylim()
        ax[i,0].set_ylim(yl)
        ax[i,1].set_ylim(yl)
        ax[i,2].set_ylim(yl)
        # Label y axes
        ax[i,0].set_ylabel(name)
    # Colorbar handling
    tickrange = np.sort(np.unique(df[colorvar]))
    cbar = fig.colorbar(cmx.ScalarMappable(norm=None, cmap=viridis),
                        ax=ax[:],
                        shrink=0.4,
                        ticks=(tickrange-colormin)/colormax)
    cbar.ax.set_yticklabels(list(map(str, tickrange)),
                            fontsize=7)
    cbar.ax.set_title(colorvar)
        


# binPlotDiff: Version of binPlot that takes pre- and post- differences.
#               Requires wbBefore be the same as wbAfter
def binPlotDiff(df,
            plotvars, groupvars, colorvar,
            numbins, wbBefore, wbAfter,
            doSTD=True,
            doSummaryStat=True):
    # Make bins
    prebin = np.linspace(0, wbBefore, numbins*wbBefore)
    stimbin = np.linspace(0, 1, numbins)
    postbin = np.linspace(0, wbAfter, numbins*wbAfter)

    # Color by delay controls
    colormax = np.max(df[colorvar])
    colormin = np.min(df[colorvar])
    # Make plot
    fig, ax = plt.subplots(len(plotvars), 4,
                           figsize=(15,10), squeeze=False,
                           gridspec_kw={'width_ratios' : [wbBefore,1,wbAfter,0.01],
                                        'wspace' : 0})
    viridis = cmx.get_cmap('viridis')
    
    # Version that takes summary stats:
    if doSummaryStat:    
        # Loop over groups
        for name, group in df.groupby(groupvars):
            # Loop over plotting variables
            for i,varname in enumerate(plotvars):
                    # Which axis to plot on, make binned means
                    # pre stim
                    if name[0]=='pre':
                        useax = 0
                        temp = group.groupby(np.digitize(group['multphase'], prebin)).agg(["mean","std"])
                    # stim
                    elif name[0]=='stim':
                        useax = 1        
                        temp = group.groupby(np.digitize(group['multphase'], stimbin)).agg(["mean","std"])
                    # post stim
                    else:
                        useax = 2
                        temp = group.groupby(np.digitize(group['multphase'], postbin)).agg(["mean","std"])
                    
                    # Plot STD shaded regions
                    if doSTD:
                        ax[i,useax].fill_between(temp['multphase']['mean'],
                                                  temp[varname]['mean'] - temp[varname]['std'],
                                                  temp[varname]['mean'] + temp[varname]['std'],
                                                  color=viridis(name[1]/colormax)[0:3],
                                                  alpha=0.5)
                    # Plot mean lines
                    ax[i,useax].plot(temp['multphase']['mean'],
                                     temp[varname]['mean'],
                                     color=viridis((name[1]-colormin)/colormax)[0:3],
                                     lw=0.5)
    # Version without summary stats
    else:
        # Loop over groups
        for name, group in df.groupby(groupvars):
            # Loop over plotting variables
            for i,varname in enumerate(plotvars):
                # Choose axis 
                if name[0]=='pre':
                    useax = 0
                elif name[0]=='stim':
                    useax = 1     
                else:
                    useax = 2
                # Plot!
                ax[i,useax].plot(group['multphase'],
                                 group[varname],
                                 color=viridis((group[colorvar].iloc[0]-colormin)/colormax)[0:3],
                                 lw=0.5)
                    
    # Axis handling    
    for i,name in enumerate(plotvars):
        # Remove yaxis labels for rightmost 2 plots
        ax[i,1].axes.get_yaxis().set_visible(False)
        ax[i,2].axes.get_yaxis().set_visible(False)
        # Set ylimits
        yl = ax[i,0].get_ylim()
        ax[i,1].set_ylim(yl)
        ax[i,2].set_ylim(yl)
        # Label y axes
        ax[i,0].set_ylabel(name)
    # Colorbar handling
    tickrange = np.sort(np.unique(df[colorvar]))
    cbar = fig.colorbar(cmx.ScalarMappable(norm=None, cmap=viridis),
                        ax=ax[:],
                        shrink=0.4,
                        ticks=(tickrange-colormin)/colormax)
    cbar.ax.set_yticklabels(list(map(str, tickrange)),
                            fontsize=7)
    cbar.ax.set_title(colorvar)
   


# wbmeanplot: Plots selected variables      
    
    