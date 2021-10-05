#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 11:13:29 2021

@author: leo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import os
import pickle
import csv
import scipy.io
from scipy.signal import butter, cheby2, filtfilt, sosfiltfilt


'''
Grab RAW data from a trial.
readMatFile below will call this to extract raw data, but apply some transformations and include some bells & whistles
'''
def readRaw(date, trial, doFT=False, readFrom='local'):
    startdir = os.path.dirname(os.path.realpath(__file__))
    if readFrom=='local': # Local data directories
        os.chdir(os.path.join(startdir, os.pardir + '/' + date))
    else: # Dropbox data directories
        os.chdir(os.path.join(os.path.expanduser('~'),
                            'Dropbox (GaTech)',
                            'Sponberg Team',
                            'Leo Wood',
                            'pitchStim',
                            date))
    # Recording type
    recname = 'FT' if doFT else 'recording'
    # Handle underscores for dates with more than 1 moth
    if '_' in date:
        date = date[:len(date)-2]
    # Set up filename
    fname = 'Moth_EMG_' + recname + '_' + date + '_' + trial
    # Load mat file
    file = scipy.io.loadmat(fname + '.mat')
    # Grab filename from first dict key that doesn't start with '_'
    fname = next((k for k in file.keys() if k[0]!='_'), None)    
    # Grab data
    datamat = file[fname]
    # Grab names 
    # (don't ask about the zeros, MANY wrapping array layers)
    names = []
    for column in file[fname+'_Header'][0][0][0][0]:
        names.append(column[0])
    # Grab sample rate
    fsamp = file[fname+'_Control'][0][0][3][0][0]
    # Move back to starting directory (assuming same as script calling this function!)
    os.chdir(startdir)
    # Return
    return datamat, names, fsamp


'''
Grab data from a trial. 
Makes some assumptions about data structure in mat file, and makes assumptions about data locations
'''
def readMatFile(date, trial,
                doFT=False, bias=np.zeros((6,1)),
                M_trans=None,
                grabOnly=[],
                readFrom='local'):
    datamat, names, fsamp = readRaw(date, trial, doFT=doFT, readFrom=readFrom)
    # If FT data, then calibrate/transform, change names
    if doFT:
        if np.any(M_trans==None):
            datamat[:,1:-1] = transformFTdata((datamat[:,1:-1]-bias).transpose())
        else:
            datamat[:,1:-1] = transformFTdata((datamat[:,1:-1]-bias).transpose(), M_trans=M_trans)
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
    return d, names, fsamp



# Grab spike-sorted data, consisting of spike times and waveforms
def readSpikeSort(date, muscles=['LDVM','LDLM','RDLM','RDVM'],
                  stimAmplitudeThresh=4,
                  readFrom='local'):
    
    startdir = os.path.dirname(os.path.realpath(__file__))
    # Jump to spikesort dir that's local
    if readFrom=='local':
        os.chdir(os.path.join(startdir, os.pardir, 'spikesort', date))
    # Jump to spikesort dir that's in dropbox
    else:
        os.chdir(os.path.join(os.path.expanduser('~'),
                              'Dropbox (GaTech)',
                              'Sponberg Team',
                              'Leo Wood',
                              'pitchStim',
                              'spikesort',
                              date))
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
            # Save placeholders for spikes and waveforms
            spikes[m] = np.array([[-1000,-1000]])
            waveforms[m] = np.zeros((1,32))
            continue
        # Determine if .mat files or .txt files were used
        fileUsed = mfiles[0][-3:]
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
            # Read in file (MAT FILE VERSION)
            if fileUsed=='mat':
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
            # Read in file (TXT FILE VERSION)
            else:
                mat = np.loadtxt(sortfile, skiprows=1, delimiter=',')
                # Loop over "channels" (actually just trials) and grab data
                for ch in np.unique(mat[:,0]):
                    # grab rows for this channel number
                    inds = mat[:,0]==ch
                    # grab spike times and put together with trial number
                    temparray = np.column_stack((mat[inds,2], mat[inds,0]))
                    # Remove any obvious stim artifacts (high amplitude!)
                    rminds = np.any(mat[inds,3:] > stimAmplitudeThresh, axis=1)
                    temparray = np.delete(temparray, (np.where(rminds)[0]), axis=0)
                    # save spike times
                    spikes[m].append(temparray)
                    # Save sort type
                    spikes[m + '_sorttype'].append(thistype)
                    # save waveforms
                    waveforms[m].append(mat[inds,3:])
        # Do a last vstack of all the nparrays in a list
        spikes[m] = np.vstack(spikes[m])
        waveforms[m] = np.vstack(waveforms[m])
    return spikes, waveforms
    # TODO: Make stimAmplitudeThresh intelligent, adjusted based on normal spike amplitudes    

# grab which trials for this moth are good and have delay
def whichTrials(date, purpose='good', readFrom='local'):
    # Move to data directory
    startdir = os.path.dirname(os.path.realpath(__file__))
    if readFrom=='local':
        os.chdir(os.path.join(startdir, os.pardir + '/' + date))
    else:
        os.chdir(os.path.join(os.path.expanduser('~'),
                              'Dropbox (GaTech)',
                              'Sponberg Team',
                              'Leo Wood',
                              'pitchStim',
                              date))
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
    # If looking for characterization trials
    if purpose=='char':
        trials = []
        # Loop over how many pairs of start-end there are
        for i in np.arange(0, len(table[1]), 2, dtype=int):
            # create list of trials
            trials.extend(np.arange(table[1][i], table[1][i+1]+1))
        return trials


'''
TRANSFORMFTDATA:
Convert raw voltages of force/torque transducer  to real force and torque values
Uses ATI calibration matrix, bias offset settings, and translation matrix of axes

Input variables:
rawData = 6 x N np matrix of gauge voltages at single time point
biasOffset = 6 x 1 np matrix of offset correction voltage values

Output Variables:
values = N x 6 matrix of transformed force/torque values in N and Nmm
'''
def transformFTdata(rawData,
                    M_trans = np.array([
                        [1,	0,	0,	0,	0,	0],
                        [0,	1,	0,	0,	0,	0],
                        [0,	0,	1,	0,	0,	0],
                        [0,	55.5,	5.4,	1,	0,	0],
                        [-55.5,	0,	0,	0,	1,	0],
                        [-5.4,	0,	0,	0,	0,	1]
                        ])
                    ):
    # Calibration Matrix, in N and N-mm:
    M_cal = np.array([
        [-0.000352378, 0.020472451, -0.02633045, -0.688977299, 0.000378075, 0.710008955],
        [-0.019191418, 0.839003543, -0.017177775, -0.37643613, 0.004482987, -0.434163392],
        [ 0.830046806, 0.004569748, 0.833562339, 0.021075403, 0.802936538, -0.001350335],
        [-0.316303442, 5.061378026, 4.614179159, -2.150699522, -4.341889297, -2.630773662],
        [-5.320003676, -0.156640061, 2.796170871, 4.206523866, 2.780562472, -4.252850011],
        [-0.056240509, 3.091367987, 0.122101875, 2.941467741, 0.005876647, 3.094672928]
        ])
    # Calibrate and Translate
    rawData = M_trans @ M_cal @ rawData
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
    sos = cheby2(order, rs, cutoff/nyq, btype=bandtype, analog=False, output='sos')
    y = sosfiltfilt(sos, signal)
    return y


def quickPlot(date, trial, tstart=5, tend=10,
              plotnames=['LDVM', 'LDLM','RDLM','RDVM'],
              hpfCutoff = 70,
              lpfCutoff = 500,
              readFrom='local',
              normalize=True):
    # Dumb hard-coded info
    channelsEMG = ['LDVM','LDLM','RDLM','RDVM']
    # Read empty data for FT
    biasmat, _, fsamp = readRaw(date, 'empty', doFT=True, readFrom=readFrom)
    bias = biasmat.mean(axis=0)
    # Read actual data
    emg, emgnames, fsamp = readMatFile(date, trial, doFT=False, readFrom=readFrom, bias=bias)
    ftd, ftdnames, _     = readMatFile(date, trial, doFT=True, readFrom=readFrom)
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
    inds = np.arange(int(tstart*fsamp), int(tend*fsamp))
    # Plot
    plt.figure()
    for i,n in enumerate(plotnames):
        yvec = full[n][inds]
        if normalize: # Rescale y to 0-1
            yvec = (yvec-np.nanmin(yvec))/(np.nanmax(yvec)-np.nanmin(yvec))
        plt.axhline(i, color='black')
        plt.plot(full['Time'][inds], yvec+i)
    # aesthetics
    plt.title(date + '-' + trial)
    plt.show()


'''
Pickle read and write convenience wrappers so I don't fuck it up
'''
def pickleWrite(varlist, filepath):
    f = open(filepath, 'wb')
    pickle.dump(varlist, f)
    f.close()
def pickleRead(filepath):
    f = open(filepath, 'rb')
    varlist = pickle.load(f)
    f.close()
    return varlist

# binPlot: More general-purpose plotting of dataframes
def binPlot(df,
            x, y, groupvars, colorvar,
            numbins, doSTD=True,
            makeplot=True):    
    # Make bins
    dx = np.diff(df[x].iloc[0:2])[0]
    bins = np.linspace(np.min(df[x]) - dx, np.max(df[x]) + dx, numbins)
    # Color by delay controls
    colormax = np.max(df[colorvar])
    colormin = np.min(df[colorvar])
    # Make plot
    if makeplot:
        fig, ax = plt.subplots(len(y), 1)
        if len(y)==1:
            ax = [ax]
        viridis = cmx.get_cmap('viridis')
    # Prep outputs
    xlist = []
    ylist = []
    glist = []
    # Loop over groups
    for name, group in df.groupby(groupvars):
        temp = group.groupby(np.digitize(group[x], bins)).agg(["mean","std"])
        ''' NOTE:
        The above code applies mean, std operation to EVERY column, including x
        This means I'm plotting the MEAN of time/phase/whatever per bin. Not wrong, but worth knowing
        '''
        # Loop over plotting variables
        for i,varname in enumerate(y):
            if makeplot:
                # Plot STD shaded regions
                if doSTD:
                    ax[i].fill_between(temp[x]['mean'],
                                       temp[varname]['mean'] - temp[varname]['std'],
                                       temp[varname]['mean'] + temp[varname]['std'],
                                       color=viridis(name/colormax),
                                       alpha=0.5)
                # Plot mean lines
                ax[i].plot(temp[x]['mean'],
                           temp[varname]['mean'],
                           color=viridis((name-colormin)/colormax),
                           lw=0.5)
            # Save
            xlist.append(temp[x])
            ylist.append(temp[varname])
            glist.append(name)
    if makeplot:
        # Axis handling
        for i,name in enumerate(y):
            # Label y axes
            ax[i].set_ylabel(name)
        # Colorbar handling
        tickrange = np.sort(np.unique(df[colorvar]))
        cbar = fig.colorbar(cmx.ScalarMappable(norm=None, cmap=viridis),
                            ax=ax[:],
                            shrink=0.4,
                            ticks=(tickrange-colormin)/colormax)
        cbar.ax.set_yticklabels(list(map(str, tickrange)),
                                fontsize=7)
        cbar.ax.set_title(colorvar)
    return xlist, ylist, glist