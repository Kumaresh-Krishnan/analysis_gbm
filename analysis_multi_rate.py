#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  pyscript.py
#  
#  Copyright 2020 Kumaresh <kumaresh_krishnan@g.harvard.edu>
#
#  version 1.0
#  

import os, sys
import numpy as np

import hdf5storage as hdf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem

import path
import pickle

from matplotlib import cm

def boutRate(raw_data, stimulus, num_bins):

    start = 'bouts_start_stimulus_%03d'%(stimulus)
    end = 'bouts_end_stimulus_%03d'%(stimulus)

    # Find bout timestamps and fish location at start
    timestamps = raw_data[start]['timestamp']
    pos_x = raw_data[start]['fish_position_x']
    pos_y = raw_data[start]['fish_position_y']

    if timestamps.size == 0:
        return np.array([np.nan]*num_bins)
    
    freq = np.full(num_bins, np.nan)
    
    for period in range(num_bins):
        
        lim1 = period*60; lim2 = lim1 + 60.0
        locs = (timestamps>lim1) & (timestamps<lim2) & (pos_x**2 + pos_y**2 < 1.0)
        normalizer = (lim2-lim1)

        freq[period] = locs.sum()

    return freq

def extractData(experiment, root):

    info_path = path.Path() / '..' / experiment

    info = np.load(info_path / 'expt_info.npy', allow_pickle=True).item()
    days = info['days']
    fish = info['fish']
    stimuli  = info['stimuli']
    folders = info['folders']
    trials = 1 #info['trials'] # One trial of free swimming no stimulus

    num_bins = 30 # 30 mins for each stimulus becomes 60 min total
    
    total_fish = np.sum(fish)
    
    fish_ctr = 0

    data = np.full((total_fish, stimuli, num_bins), np.nan)

    for day_idx, day in enumerate(days):

        for f in range(fish[day_idx]):

            folder = root / folders[day_idx] / f'{day}_fish{f+1:03d}' / 'raw_data' / f'trial000.dat'

            tmp = open(folder, 'rb')
            raw_data = pickle.load(tmp)

            order = sorted(list(range(stimuli)), key=lambda x: raw_data[f'raw_stimulus_{x:03d}']['camera_framenum'][0])
            for stimulus in range(stimuli):
                    
                rate = boutRate(raw_data, stimulus, num_bins)
                data[fish_ctr, order[stimulus]] = rate

            tmp.close()

            fish_ctr += 1
                    
        print(day, fish_ctr, 'fish done')

    # Return a reshaped array with stimuli concatenated since it is 1 hr experiment
    
    return data.reshape(data.shape[0], -1)

def processData(experiment, data):

    info_path = path.Path() / '..' / experiment
    info = np.load(info_path / 'expt_info.npy', allow_pickle=True).item()

    groups = info['groups']

    to_save = {}

    for group in groups.keys():

        freq = np.nanmean(data[groups[group]], axis=0)
        sem_freq = sem(data[groups[group]], axis=0, nan_policy='omit')

        to_save[f'freq_{group}'] = freq
        to_save[f'sem_{group}'] = sem_freq.data if np.ma.isMaskedArray(sem_freq) else sem_freq
        to_save[f'raw_{group}'] = data[groups[group]]

    return to_save

def plotHistogram(experiment):

    data_path = path.Path() / '..' / experiment
    tmp = hdf.loadmat(data_path / f'data_rate')

    info = np.load(data_path / 'expt_info.npy', allow_pickle=True).item()
    groups = info['groups']

    save_dir = path.Path() / '..' / experiment

    sns.set_style('white')
    sns.set_style('ticks') 

    f, ax = plt.subplots()
    
    for group in groups.keys():
        
        data = tmp[f'freq_{group}']
        sem_data = tmp[f'sem_{group}']

        times = np.arange(data.shape[0])

        ax.plot(times, data, label=group)
        plt.fill_between(times, data-sem_data, data+sem_data, color='gray', alpha=0.5)
        
    ax.set_xlabel(f'Time (min)')
    ax.set_ylabel(f'Boute rate (bouts/min)')
    ax.set_title(f'Bout rate through experiment')
    ax.set_ylim(0,100)
    ax.legend()
    
    ax.grid(False)
    sns.despine(top=True, right=True)

    f.savefig(save_dir / f'fig_activity.pdf')
    f.savefig(save_dir / f'fig_activity.png')
    plt.close(f)

    return 0

def main(experiment):

    root = path.Path() / '..' / '..'
    
    data = extractData(experiment, root,)

    to_save = processData(experiment, data)

    save_dir = path.Path() / '..' / experiment / f'data_rate'
    hdf.savemat(save_dir, to_save, format='7.3', oned_as='column', store_python_metadata=True)

    return 0

if __name__ == '__main__':

    #experiment = 'stripe_baseline_ketamine_08_31_2022'
    experiment = 'correct_riluzole_04_16_2022'

    #main(experiment)
    plotHistogram(experiment)

    sys.exit(0)
