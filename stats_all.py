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
import csv

import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns

import hdf5storage as hdf
import path

def rate24(dpath):

    tmp = hdf.loadmat(dpath / 'data_rate.mat')
    data_ctrl = tmp['raw_ctrl']
    data_gbm = tmp['raw_gbm']
    data_mod_ctrl = tmp['raw_gbm']
    data_mod_gbm = tmp['raw_r10_gbm']

    m_ctrl = np.nanmean(data_ctrl, axis=1)
    m_gbm = np.nanmean(data_gbm, axis=1)
    m_mod_ctrl = np.nanmean(data_mod_ctrl, axis=1)
    m_mod_gbm = np.nanmean(data_mod_gbm, axis=1)
    valids = ~np.isnan(m_mod_gbm)

    # Only for cnqx, had to check how many nans are present

    stats_plain = ss.ttest_ind(m_ctrl, m_gbm)
    stats_mod = ss.ttest_ind(m_mod_ctrl, m_mod_gbm)
    stats_rmod = ss.ttest_ind(m_ctrl, m_mod_gbm)
    print(stats_plain, stats_rmod, stats_mod)

    return 0

def turningStats2(dpath):

    eps = 1e-5
    
    tmp = hdf.loadmat(dpath / 'data_72.mat')
    data_ctrl = tmp['prob_ctrl']
    data_gbm = tmp['prob_gbm']
    data_mod_ctrl = tmp['prob_cmqx_ctrl']
    data_mod_gbm = tmp['prob_cmqx_gbm']

    dbl_ctrl = (data_ctrl[1] + np.flip(data_ctrl[0])) / 2. + eps
    dbl_gbm = (data_gbm[1] + np.flip(data_gbm[0])) / 2. + eps
    dbl_mod_ctrl = (data_mod_ctrl[1] + np.flip(data_mod_ctrl[0])) / 2. + eps
    dbl_mod_gbm = (data_mod_gbm[1] + np.flip(data_mod_gbm[0])) / 2. + eps
    val = 16
    dbl_ctrl = dbl_ctrl[val:-val]
    dbl_mod_ctrl = dbl_mod_ctrl[val:-val]
    dbl_gbm = dbl_gbm[val:-val]
    dbl_mod_gbm = dbl_mod_gbm[val:-val]
    # print(dbl_ctrl)
    dbl_ctrl = dbl_ctrl / dbl_ctrl.sum()
    dbl_gbm = dbl_gbm / dbl_gbm.sum()
    dbl_mod_ctrl = dbl_mod_ctrl / dbl_mod_ctrl.sum()
    dbl_mod_gbm = dbl_mod_gbm / dbl_mod_gbm.sum()
    print(dbl_ctrl.sum())
    mix1 = (dbl_ctrl + dbl_gbm) * 0.5
    mix2 = (dbl_ctrl + dbl_mod_gbm) * 0.5
    mix3 = (dbl_mod_ctrl + dbl_mod_gbm) * 0.5
    #print(dbl_gbm, dbl_ctrl)#; input()
    kl_plain = 0.5*(dbl_gbm * np.log(dbl_gbm / dbl_ctrl)).sum() + \
        0.5*(dbl_ctrl * np.log(dbl_ctrl / dbl_gbm)).sum()
    kl_1mod = 0.5*(dbl_mod_gbm * np.log(dbl_mod_gbm / dbl_ctrl)).sum() + \
        0.5*(dbl_ctrl * np.log(dbl_ctrl / dbl_mod_gbm)).sum()
    kl_2mod = 0.5*(dbl_mod_gbm * np.log(dbl_mod_gbm / dbl_mod_ctrl)).sum() + \
        0.5*(dbl_mod_ctrl * np.log(dbl_mod_ctrl / dbl_mod_gbm)).sum()
    print(kl_plain, kl_1mod, kl_2mod)

    return

if __name__ == '__main__':

    experiment = 'stripe_baseline_cmqx_08_31_2022'
    experiment = 'correct_riluzole_04_16_2022'

    dpath = path.Path() / '..' / experiment

    rate24(dpath)
    #turningStats2(dpath)

    sys.exit()
