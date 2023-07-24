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

def headingAngle(raw_data, stimulus, num_bins):

    start = 'bouts_start_stimulus_%03d'%(stimulus)
    end = 'bouts_end_stimulus_%03d'%(stimulus)
    
    # Compute differences (convention uses start-end)
    angles = raw_data[start]['fish_accumulated_orientation'] - \
                       raw_data[end]['fish_accumulated_orientation']

    if angles.size == 0:
        return np.array([np.nan]*num_bins)

    # Find bout timestamps and fish location at start
    timestamps = raw_data[start]['timestamp']
    pos_x = raw_data[start]['fish_position_x']
    pos_y = raw_data[start]['fish_position_y']

    # Filter angles based on stimulus time and distance from edge of dish
    # Normalize by exposure to stimulus and suitable scaling for numerical value
    scale = 1

    lim1, lim2 = 5.0, 15.00
    locs = np.where((timestamps>lim1) & (timestamps<lim2) & (pos_x**2 + pos_y**2 < 0.81))
    normalizer = (lim2-lim1)/scale

    angles = angles[locs]

    # Restrict range to -180,180 and compute frequencies with specified num_bins
    angles[angles > 180] -= 360
    freq, _ = np.histogram(angles, bins=num_bins, range=(-180,180))

    return freq/normalizer

def extractAngles(experiment,root, num_bins):

    info_path = path.Path() / '..' / experiment

    info = np.load(info_path / 'expt_info.npy', allow_pickle=True).item()

    days = info['days']
    fish = info['fish']
    trials = info['trials']
    folders = info['folders']
    total_fish = np.sum(fish)

    fish_ctr = 0
    stimuli = info['stimuli']

    data = np.full((total_fish, trials-1, stimuli, num_bins), np.nan)

    for day_idx, day in enumerate(days):

        for f in range(fish[day_idx]):

            for t in range(1,trials): # trials are 1 to 31

                folder = root / folders[day_idx] / f'{day}_fish{f+1:03d}' / 'raw_data' / f'trial{t:03d}.dat'

                tmp = open(folder, 'rb')
                raw_data = pickle.load(tmp)

                for stimulus in range(stimuli):
                    
                    angles = headingAngle(raw_data, stimulus, num_bins)
                    data[fish_ctr, t-1, stimulus] = angles # trials are 1 to 31

                    tmp.close()

            fish_ctr += 1
                    
        print(day, fish_ctr, 'fish done')
        
    return data

def processAngles(experiment, data, num_bins):

    info_path = path.Path() / '..' / experiment
    info = np.load(info_path / 'expt_info.npy', allow_pickle=True).item()

    groups = info['groups']
    
    to_save = {}

    for group in groups.keys():

        tmp_data = np.nanmean(data[groups[group]], axis=1)

        avg_data = np.nanmean(tmp_data, axis=0)
        sem_data = sem(tmp_data, axis=0, nan_policy='omit')

        norm_data = avg_data.sum(axis=1).reshape(-1,1)
        prob_data = avg_data / norm_data
        sem_prob = sem_data / norm_data

        tmp_sum = np.nansum(data[groups[group]], axis=3)
        tmp_freq = np.nanmean(tmp_sum, axis=1)
        freq_data = np.nanmean(tmp_freq, axis=0)
        sem_freq = sem(tmp_freq, axis=0, nan_policy='omit')

        to_save[f'mean_{group}'] = avg_data
        to_save[f'sem_{group}'] = sem_data.data if np.ma.isMaskedArray(sem_data) else sem_data
        to_save[f'prob_{group}'] = prob_data
        to_save[f'sem_prob_{group}'] = sem_prob.data if np.ma.isMaskedArray(sem_prob) else sem_prob
        to_save[f'freq_{group}'] = freq_data
        to_save[f'sem_freq_{group}'] = sem_freq.data if np.ma.isMaskedArray(sem_freq) else sem_freq
        to_save[f'raw_{group}'] = tmp_freq
        to_save[f'raw_ang_{group}'] = tmp_data

    return to_save

def plotHistogram(experiment, num_bins, prob=False):

    data_path = path.Path() / '..' / experiment 
    tmp = hdf.loadmat(data_path / f'data_{num_bins}')

    info = np.load(data_path / 'expt_info.npy', allow_pickle=True).item()
    groups = info['groups']
    stimuli = info['stimuli']

    sns.set_style('white')
    sns.set_style('ticks')

    angles = np.linspace(-180,180, num_bins)
    id_map = {'0': 'Leftward', '1': 'Rightward', '2': 'Baseline'}

    if prob:
        save_dir = data_path / f'stimulus_histograms_{num_bins}_prob'
        save_dir_db = data_path / f'doubled_stimulus_histograms_{num_bins}_prob'
    else:
        save_dir = data_path / f'stimulus_histograms_{num_bins}'
        save_dir_db = data_path / f'doubled_stimulus_histograms_{num_bins}'

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_db, exist_ok=True)

    for stimulus in range(stimuli):

        f, ax = plt.subplots()

        for group in groups.keys():

            if prob:
                data = tmp[f'prob_{group}']
                sem_data = tmp[f'sem_prob_{group}']

            else:
                data = tmp[f'mean_{group}']
                sem_data = tmp[f'sem_{group}']

            ax.plot(angles, data[stimulus], label=group)
            
            ax.fill_between(angles, \
                data[stimulus]-sem_data[stimulus], \
                data[stimulus]+sem_data[stimulus], \
                color='gray', alpha=0.5)

        ax.set_xlabel(f'$\\Delta$ Angle (°)')
        if prob == 1:
            ax.set_ylabel(f'Probability')
            ax.set_ylim(0,0.22)
        else:
            ax.set_ylabel(f'Frequency (Hz)')
            ax.set_ylim(0,0.22)
        ax.set_title(f'{id_map[str(stimulus)]} Stimulus, (Bin: 5$^\circ$)')
        ax.legend()
        
        ax.grid(False)
        sns.despine(top=True, right=True)

        f.savefig(save_dir / f'fig_{stimulus}_{id_map[str(stimulus)]}.pdf')
        f.savefig(save_dir / f'fig_{stimulus}_{id_map[str(stimulus)]}.png')
        plt.close(f)

    half = stimuli // 2

    for stimulus in range(half):

        f, ax = plt.subplots()

        for group in groups.keys():

            if prob:
                data = tmp[f'prob_{group}']
                sem_data = tmp[f'sem_prob_{group}']

            else:
                data = tmp[f'mean_{group}']
                sem_data = tmp[f'sem_{group}']

            data = (np.fliplr(data[:half]) + data[half:]) / 2.0
            sem_data = np.sqrt((np.fliplr(sem_data[:half])**2 + sem_data[half:]**2) / 2.0)

            correct = data[stimulus,36:].sum() / data[stimulus].sum()
            
            ax.plot(angles, data[stimulus], label=f'{group} {correct:.2f}')
            
            ax.fill_between(angles, \
                data[stimulus]-sem_data[stimulus], \
                data[stimulus]+sem_data[stimulus], \
                color='gray', alpha=0.5)

            ax.set_xlabel(f'$\\Delta$ Angle (°)')
            if prob == 1:
                ax.set_ylabel(f'Probability')
                ax.set_ylim(0,0.22)
            else:
                ax.set_ylabel(f'Frequency (Hz)')
                ax.set_ylim(0,0.22)
            ax.set_title(f'{id_map[str(half+stimulus)]} Stimulus, (Bin: 5$^\circ$)')
            ax.legend()
            
            ax.grid(False)
            sns.despine(top=True, right=True)

            f.savefig(save_dir_db / f'fig_{stimulus}_{id_map[str(half+stimulus)]}.pdf')
            f.savefig(save_dir_db / f'fig_{stimulus}_{id_map[str(half+stimulus)]}.png')
            plt.close(f)

    return 0

def boutFrequency(experiment, num_bins):

    data_path = path.Path() / '..' / experiment 
    tmp = hdf.loadmat(data_path / f'data_{num_bins}')

    info = np.load(data_path / 'expt_info.npy', allow_pickle=True).item()

    groups = info['groups']
    stimuli = info['stimuli']

    id_map = hdf.loadmat(data_path / 'ID_map.mat')

    save_dir = path.Path() / '..' / experiment
    
    x_range = range(stimuli)

    sns.set_style('white')
    sns.set_style('ticks')

    f, ax = plt.subplots()

    space = 1.

    for group in groups.keys():

        freq = tmp[f'freq_{group}']
        sem_freq = tmp[f'sem_freq_{group}']
        raw_freq = tmp[f'raw_{group}']

        ax.bar([e + space for e in list(x_range)], freq, yerr=sem_freq, \
            capsize=5.0, label=group, alpha=0.5, width=1/len(groups.keys()))
    
        for i in x_range:
            x = [i+space] * raw_freq.shape[0]
            ax.scatter(x, raw_freq[:,i], color = 'grey')

        space += 1 / len(groups.keys())
        
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('Total number of bouts')
    ax.set_title('Total response to stimulus')
    ax.set_xticks([i+1 for i in x_range])
    text = [str(x+1) for x in x_range]
    ax.set_xticklabels(text)
    ax.legend()
    ax.grid(False)
    
    sns.despine(top=True, right=True)

    f.savefig(save_dir / f'fig_total_response.pdf')
    f.savefig(save_dir / f'fig_total_response.png')
    plt.close(f)

    half = stimuli // 2
    
    x_range = range(half)
    
    f, ax = plt.subplots()

    space = 1.

    for group in groups.keys():

        raw_freq = tmp[f'raw_{group}']

        raw_freq = (raw_freq[:,:half] + raw_freq[:,half:]) / 2.0
        freq = np.nanmean(raw_freq, axis=0)
        sem_freq = sem(raw_freq, axis=0, nan_policy='omit')

        ax.bar([e + space for e in list(x_range)], freq, yerr=sem_freq, \
            capsize=5.0, label=group, alpha=0.5, width=1/len(groups.keys()))

        for i in x_range:
            x = [i+space] * raw_freq.shape[0]
            ax.scatter(x, raw_freq[:,i], color = 'grey')

        space += 1/len(groups.keys())

    ax.set_xlabel('Stimulus')
    ax.set_ylabel('Total bout rate')
    ax.set_title('Total response to stimulus')
    ax.set_xticks([i+1 for i in x_range])
    text = [str(x+1) for x in x_range]
    ax.set_xticklabels(text)
    ax.legend()
    ax.grid(False)

    sns.despine(top=True, right=True)

    f.savefig(save_dir / f'fig_total_response_doubled.pdf')
    f.savefig(save_dir / f'fig_total_response_doubled.png')
    plt.close(f)

def main(experiment, num_bins):

    #root = path.Path() / '..' / '..' / '..' / 'data_hanna_test_06_16_2021' # directory for data
    root = path.Path() / '..' / '..'
    #root = path.Path('/media/kumaresh/T7/Behavior Experiments/d1_free_stripes_03_02_2022')
    
    data = extractAngles(experiment, root, num_bins)

    to_save = processAngles(experiment, data, num_bins)

    save_dir = path.Path() / '..' / experiment / f'data_{num_bins}'
    hdf.savemat(save_dir, to_save, format='7.3', oned_as='column', store_python_metadata=True)

    return 0

if __name__ == '__main__':

    #experiment = 'stripe_baseline_ketamine_08_31_2022'
    experiment = 'correct_riluzole_04_16_2022'
    num_bins = 72

    #main(experiment, num_bins)
    plotHistogram(experiment, num_bins)
    plotHistogram(experiment, num_bins, True)
    #boutFrequency(experiment, num_bins)

    sys.exit(0)
