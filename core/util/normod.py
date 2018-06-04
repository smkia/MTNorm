# -*- coding: utf-8 -*-
"""
Created on Wed Aug 02 16:26:41 2017

@author: Mosi
"""
import numpy as np
from scipy.stats import genextreme, trim_mean


def normative_prob_map(Y, Y_hat, S_hat_2, S_n_2):
    NPM = (Y - Y_hat) / np.sqrt(S_hat_2 + np.repeat(np.expand_dims(S_n_2,1).T, Y.shape[0], axis = 0))
    return NPM
    
def extreme_value_prob_fit(NPM, perc):
    n = NPM.shape[0]
    t = NPM.shape[1]
    n_perc = int(round(t * perc))
    m = np.zeros(n)
    for i in range(n):
        temp =  np.abs(NPM[i, :])
        temp = np.sort(temp)
        temp = temp[t - n_perc:]
        m[i] = trim_mean(temp, 0.05)
    params = genextreme.fit(m)
    return params
    
def extreme_value_prob(params, NPM, perc):
    n = NPM.shape[0]
    t = NPM.shape[1]
    n_perc = int(round(t * perc))
    m = np.zeros(n)
    for i in range(n):
        temp =  np.abs(NPM[i, :])
        temp = np.sort(temp)
        temp = temp[t - n_perc:]
        m[i] = trim_mean(temp, 0.05)
    probs = genextreme.cdf(m,*params)
    return probs
    