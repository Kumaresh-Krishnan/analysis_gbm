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
import hdf5storage as hdf
import pickle
import path

import matplotlib.pyplot as plt
import seaborn as sns

def mainBleh(day):

    f = plt.figure()

    
    for t in range(50):
        start = 'bouts_start_stimulus_000'
        end = 'bouts_end_stimulus_000'

        fish = 1
        fname = path.Path() / day + f'_fish{fish:03d}'
        fl = open(fname / 'raw_data' / f'trial{t:03d}.dat', 'rb')
        tmp = pickle.load(fl)
        
        acc = tmp[end]['fish_accumulated_orientation']
        ts = tmp[end]['timestamp']
        plt.plot(ts, acc, marker='o')

    plt.axvspan(0,0.5, color='grey', alpha=0.5)
    plt.axvspan(0.5,1.0, edgecolor='grey', facecolor='white', alpha=0.8)
    plt.xlim([0,5])
    plt.xlabel('Time')
    plt.ylabel('Angle')
    plt.title('Acc orientation')
    plt.grid(False)

    plt.show()
    
    
    return 0

def main(expt, day):

    f, ax1 = plt.subplots()
    g, ax2 = plt.subplots()


    for fish in range(1,17):

        record = np.zeros(60)
        idx = 0
        nope = 0
        absent = 0
    
        for t in range(0,60):
            #print(t, fish)
            start = 'bouts_start_stimulus_000'
            end = 'bouts_end_stimulus_000'


            fname = expt / day + f'_fish{fish:03d}'
            fl = open(fname / 'raw_data' / f'trial{t:03d}.dat', 'rb')
            tmp = pickle.load(fl)
            
            ang_start = tmp[start]['fish_accumulated_orientation']
            ts = tmp[start]['timestamp']
            
            if len(ang_start) == 0 or np.isnan(ang_start[0]):
                record[idx] = 1; idx += 1; nope += 1
                continue

            change = np.abs(np.diff(ang_start))
            change_change = np.abs(np.diff(change))
            
            if change.size < 2 or (change[0] < 50 and change[1] < 50):
                record[idx] = 2; idx += 1; absent += 1
                continue
            
            filt =  (change[0] > 100) & (change <= 200); filt_2 = (change_change[0] > 100) & (change_change <= 200)
            ax1.plot(ts[:-1][filt], change[filt], marker='o', color='#2ca02c', alpha=0.99)
            ax2.plot(ts[:-2][filt_2], change_change[filt_2], marker='o', color='#2ca02c', alpha=0.99) # 1f7bb4, 2cca02c
            idx += 1
        
        #print(f'Fish {fish}: {record}')
        print(f'Fish {fish}: Absent: {absent} No: {nope} Yes: {60-nope-absent} Score: {(60-nope-absent)/60*100:.2f}% Adj: {(60-nope-absent)/(60-absent)*100:.2f} Miss: {absent/60:.2f}')

        
    ax1.axvline(0.5, color='black', linestyle='--')
    ax1.axvline(1.5, color='black', linestyle='--')
    ax2.axvline(0.5, color='black', linestyle='--')
    ax2.axvline(1.5, color='black', linestyle='--')
    
    ax1.axvspan(0,0.5, edgecolor='black', facecolor='grey', alpha=0.5, ls='--')
    ax2.axvspan(0,0.5, edgecolor='black', facecolor='grey', alpha=0.5, ls='--')
    ax1.axvspan(0.5,1.5, edgecolor='black', facecolor='white', alpha=0.5, ls='--')
    ax2.axvspan(0.5,1.5, edgecolor='black', facecolor='white', alpha=0.5, ls='--')
    ax1.set_xlim([0,5]); ax2.set_xlim([0,5])
    ax1.set_ylim([0, 250]); ax2.set_ylim([0, 250])
    ax1.set_xlabel('Time'); ax2.set_xlabel('Time')
    ax1.set_ylabel('Angle'); ax2.set_ylabel('Angle')
    ax1.set_title('Angle change between bout[i] bout[i-1])')
    ax2.set_title('Change in angle change')

    ax1.grid(False); ax2.grid(False)

    plt.show()


    return 0

if __name__ == '__main__':

    expt = path.Path() / '..' / '..' / 'data_gbm_flash' 
    day = '2021_11_25'
    
    main(expt, day)
    sys.exit()
