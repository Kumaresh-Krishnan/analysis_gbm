#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  pyscript.py
#  
#  Copyright 2019 Kumaresh <kumaresh_krishnan@g.harvard.edu>
#  
#  version 1.0

import numpy as np
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
import path
import hdf5storage as hdf

def main(experiment):

    data_path = path.Path() / '..' / experiment

    tmp_angle = hdf.loadmat(data_path / 'data_72')
    tmp_rate = hdf.loadmat(data_path / 'data_rate')
    info = np.load(data_path / 'expt_info.npy', allow_pickle=True).item()

    groups = info['groups']

    save_dir = data_path / 'comparison_plots'
    os.makedirs(save_dir, exist_ok=True)


    pairs = [('ctrl', 'r10_ctrl'), ('293', 'r10_293'), ('gbm', 'r10_gbm'), \
        ('ctrl', 'gbm'), ('r10_ctrl', 'r10_gbm'), ('ctrl', '293'), \
        ('r10_ctrl', 'r10_293'), ('gbm', '293'), ('r10_gbm', 'r10_293')]

    triples = [('ctrl', '293', 'gbm'), ('r10_ctrl', 'r10_293', 'r10_gbm')]

    for p0, p1 in pairs:
         
        plotPairAngle(tmp_angle[f'mean_{p0}'], tmp_angle[f'mean_{p1}'], \
            tmp_angle[f'sem_{p0}'], tmp_angle[f'sem_{p1}'], \
            p0, p1, save_dir)

        plotPairRate(tmp_rate[f'freq_{p0}'], tmp_rate[f'freq_{p1}'], \
            tmp_rate[f'sem_{p0}'], tmp_rate[f'sem_{p1}'], \
            p0, p1, save_dir)

    for p0, p1, p2 in triples:
         
        plotPairAngle(tmp_angle[f'mean_{p0}'], tmp_angle[f'mean_{p1}'], \
            tmp_angle[f'sem_{p0}'], tmp_angle[f'sem_{p1}'], \
            p0, p1, save_dir, \
            tmp_angle[f'mean_{p2}'], tmp_angle[f'sem_{p2}'], p2)

        plotPairRate(tmp_rate[f'freq_{p0}'], tmp_rate[f'freq_{p1}'], \
            tmp_rate[f'sem_{p0}'], tmp_rate[f'sem_{p1}'], \
            p0, p1, save_dir, \
            tmp_rate[f'freq_{p2}'], tmp_rate[f'sem_{p2}'], p2)
    
    return 0

def plotPairAngle(d1, d2, s1, s2, l1, l2, save_dir, d3=[], s3=None, l3=None):

    x = np.linspace(-180,180,72)

    d1 = (np.flip(d1[0]) + d1[1]) / 2
    d2 = (np.flip(d2[0]) + d2[1]) / 2

    s1 = np.sqrt(np.flip(s1[0])**2 + s1[1]**2)
    s2 = np.sqrt((np.flip(s2[0])**2 + s2[1]**2) / 2)

    if len(d3) != 0:
        d3 = (np.flip(d3[0]) + d3[1]) / 2
        s3 = np.sqrt((np.flip(s3[0])**2 + s3[1]**2) / 2)
    
    sns.set_style('white')
    sns.set_style('ticks')
    
    f, ax = plt.subplots()

    c1 = d1[36:].sum() / d1.sum()
    c2 = d2[36:].sum() / d2.sum()

    ax.plot(x, d1, label=f'{l1} {c1:.2f}')
    ax.plot(x, d2, label=f'{l2} {c2:.2f}')

    ax.fill_between(x, d1-s1, d1+s1, color='grey', alpha=0.5)
    ax.fill_between(x, d2-s2, d2+s2, color='grey', alpha=0.5)

    if len(d3) != 0:
        c3 = d3[36:].sum() / d3.sum()
        ax.plot(x, d3, label=f'{l3} {c3:.2f}')
        ax.fill_between(x, d3-s3, d3+s3, color='grey', alpha=0.5)
        ax.set_title(f'Comparing {l1}, {l2} and {l3}')
    else:
        ax.set_title(f'Comparing {l1} and {l2}')

    ax.set_xlabel(f'Angle ($^\circ$)')
    ax.set_ylabel(f'Frequency (Hz)')
    ax.set_ylim(0,0.22)
    ax.legend()
    ax.grid(False)
    sns.despine(top=True, right=True)

    f.savefig(save_dir / f'{l1}_{l2}_{l3}_angle.pdf')
    plt.close(f)

    return

def plotPairRate(d1, d2, s1, s2, l1, l2, save_dir, d3=[], s3=None, l3=None):

    x = np.arange(d1.shape[0])
    
    sns.set_style('white')
    sns.set_style('ticks')
    
    f, ax = plt.subplots()

    c1 = np.nanmean(d1)
    c2 = np.nanmean(d2)

    ax.plot(x, d1, label=f'{l1} {c1:.2f}')
    ax.plot(x, d2, label=f'{l2} {c2:.2f}')

    ax.fill_between(x, d1-s1, d1+s1, color='grey', alpha=0.5)
    ax.fill_between(x, d2-s2, d2+s2, color='grey', alpha=0.5)

    if len(d3) != 0:
        c3 = np.nanmean(d3)
        ax.plot(x, d3, label=f'{l3} {c3:.2f}')
        ax.fill_between(x, d3-s3, d3+s3, color='grey', alpha=0.5)
        ax.set_title(f'Comparing {l1}, {l2} and {l3}')
    else:
        ax.set_title(f'Comparing {l1} and {l2}')

    ax.set_xlabel(f'Time (min)')
    ax.set_ylabel(f'Frequency (Bouts per min)')
    ax.set_ylim(0,100)
    ax.legend()
    ax.grid(False)
    sns.despine(top=True, right=True)

    f.savefig(save_dir / f'{l1}_{l2}_{l3}_rate.pdf')
    plt.close(f)

    return f


if __name__ == '__main__':

    experiment = 'correct_riluzole_04_16_2022'
    
    main(experiment)
    sys.exit()
