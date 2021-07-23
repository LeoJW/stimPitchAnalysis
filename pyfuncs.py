#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 11:13:29 2021

@author: leo
"""

import numpy as np
import matplotlib.pyplot as plt
import time as systime
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
    table = []
    reader = csv.reader(open(guide[0]))
    for row, record in enumerate(reader):
        table.append(record)
    # if looking for good delay trial
    if purpose=='good':
        # Grab first good delay trial
        start = int(table[2][1])
        # grab last good delay trial
        if len(table[3][1])==0:
            end = int(table[2][2])
        else:
            end = int(table[3][1])-1
        return [start, end]
    # Can extend to return other types of trials (stim characterization) eventually
    
    
    


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
Create binned summary stat for list of x and y vectors 
TODO: Write actually good documentation lol
'''
# def summaryStatBin(xlist, ylist, bins=100, stat):
    



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
                               plotnames=['LDVM', 'LDLM','RDLM','RDVM']):
    # Read empty data for FT
    biasdata, colnames, fsamp = readMatFile(date, 'empty', doFT=True)
    bias = np.zeros((6,1))
    for i in range(6):
        bias[i] = np.mean(biasdata[colnames[i+1]])
    # Read actual data
    emg, emgnames, fsamp = readMatFile(date, trial, doFT=False)
    ftd, ftdnames, _     = readMatFile(date, trial, doFT=True)
    # Find indices to plot
    inds = np.arange(tstart*fsamp, tend*fsamp)
    # Plot
    plt.figure()
    for i,n in enumerate(plotnames):
        # Rescale y to 0-1
        # TODO: Rescaling not working! need to ignore stim artifact
        yvec = emg[n][inds]
        yvec = yvec*(np.max(yvec)-np.min(yvec))
        plt.plot(emg['Time'][inds], yvec+i)
    plt.show()

        
        