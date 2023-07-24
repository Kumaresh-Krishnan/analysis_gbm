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

    data_path = path.Path() / '..' / '..' / experiment

    tmp = hdf.loadmat(data_path / 'data_72')
    info = np.load(data_path / 'expt_info.npy', allow_pickle=True).item()

    groups = info['groups']
    
    
    return 0

def plotSet(data, labels):

    angles = 

    return

if __name__ == '__main__':

    experiment = 'stripe_baseline_r10_recovery_04_09_2022'
    
    main(experiment)
    sys.exit()
