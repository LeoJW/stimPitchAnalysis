#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 13:35:48 2021

Convert to format for offline sorter to use

@author: leo
"""

import numpy as np
import scipy.io 
import os


sortChannels = ['LDVM','LDLM','RDLM','RDVM']


# Move back one directory
startdir = os.path.dirname(os.path.realpath(__file__))
os.chdir(os.path.join(startdir, '..'))
# Grab all directories with "202107" in name
outerdir = os.getcwd()
folders = os.listdir()
trialdirs = [s for s in folders if '202107' in s]




# Loop over trial directories
for dircount,d in enumerate(trialdirs):
    # Jump to this trial
    os.chdir(os.path.join(outerdir, d))
    
    # Print to let user know which dir it's on
    print('\n')
    print(d)
    print('   ' + str(dircount+1) + '/' + str(len(trialdirs)), end=' ')
    
    # Check if spikesort folder exists, make it if not
    if not any('spikesort' in s for s in os.listdir()):
        os.mkdir('spikesort')
    
    # Get names of all .mat files (that aren't empty or Control)
    mfiles = [s for s in os.listdir()
              if '.mat' in s 
              if 'empty' not in s 
              if 'Control' not in s
              if 'FT' not in s]
    
    # Reorder so trial numbers are in the right order 001->increasing
    # Get correct order
    neworder = []
    for f in mfiles:
        neworder.append(int(f[-7:-4]) - 1)
    # Rearrange
    mfiles = [mfiles[i] for i in np.argsort(neworder)]
    
    # Preallocate arrays to save for each channel
    final = {}
    for x in sortChannels:
        final[x] = np.zeros((len(mfiles), 20*10000))
    
    # Loop over mat files
    for ii,f in enumerate(mfiles):
        # Read this mat file
        mat = scipy.io.loadmat(f)
        # Grab filename from first dict key that doesn't start with '_'
        fname = next((k for k in mat.keys() if k[0]!='_'), None)    
        # Grab data
        datamat = mat[fname]
        # Grab names 
        # (don't ask about the zeros, MANY wrapping array layers)
        names = []
        for column in mat[fname+'_Header'][0][0][0][0]:
            names.append(column[0])
        
        # Loop over each channel to save
        for ch in sortChannels:
            # Find which column has data for this channel
            col = [i for i,x in enumerate(names) if x==ch]
            # Transpose and place into row of final array
            final[ch][ii,0:np.shape(datamat[:,col])[0]] = np.transpose(datamat[:,col])
        
    # Jump into spikesort folder
    os.chdir(os.path.join(outerdir, d, 'spikesort'))
    # Save each channel as its own mat file
    for ch in sortChannels:
        scipy.io.savemat(ch+'_raw.mat', {'file' : final[ch]})
    