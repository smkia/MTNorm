# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 09:25:02 2017

@author: Mosi
"""
import scipy as SP
import numpy as np
import pandas as pd
import os
from core.util import fileio


def read_phenomics_data(main_dir, diagnosis = 'CONTROL', task_name = 'stopsignal', cope_num = 1, phenotypes = ['stopsignal'], mask = 'general', vol = True, scanner=None):
    subjects_info = pd.read_csv(main_dir + 'participants.tsv', delimiter='\t')
    if scanner==None:
        subj_list =  subjects_info.loc[lambda subjects_info: subjects_info.diagnosis==diagnosis,'participant_id']
    else:
        a = np.asarray(subjects_info.diagnosis==diagnosis) * np.asarray(subjects_info.ScannerSerialNumber==scanner)
        subj_list =  subjects_info.loc[lambda subjects_info: a,'participant_id']
    phenotype_info = list()
    for i in range(len(phenotypes)):
        temp = pd.read_csv(main_dir + 'phenotype/' + phenotypes[i] + '.tsv', delimiter='\t')
        if i == 0:      
            temp = temp.iloc[:,1:]
        else:
            temp = temp.iloc[:,2:]
        phenotype_info.append(temp)
    phenotype_info = pd.concat(phenotype_info,axis=1)
    fmri_data = list()
    phenotype_data = list()
    for i in range(len(subj_list)):
        address = main_dir + 'derivatives/task/' +  subj_list.iloc[i] + '/' + task_name + '.feat/stats/' + 'cope' + str(cope_num) + '.nii.gz'
        if os.path.isfile(address):
            volume = fileio.load_nifti(address, vol = True)
            if mask == 'general':
                volmask = fileio.create_mask(volume, mask = main_dir + '/derivatives/task_group/' + task_name + '/mask.nii.gz')
            elif mask == 'group':              
                volmask = fileio.create_mask(volume, mask = main_dir + '/derivatives/task_group/' + task_name + '/cope' + str(cope_num) + '/flame1/mask.nii.gz')
            if vol:
                volume = volume * volmask
                fmri_data.append(volume)   
            else:
                fmri_data.append(fileio.vol2vec(volume, volmask))
            temp = phenotype_info.loc[lambda phenotype_info: phenotype_info.participant_id==subj_list.iloc[i],:]
            temp = temp.apply(pd.to_numeric, errors='coerce')
            phenotype_data.append(np.float32(np.asarray(temp.iloc[0,1:])))
    return np.asarray(fmri_data), np.asarray(phenotype_data)

def ravel(Y):
    """
    returns a flattened array: columns are concatenated
    """
    return SP.ravel(Y,order='F')

def unravel(Y,n,t):
    """
    returns a nxt matrix from a raveled matrix
    Y = uravel(ravel(Y))
    """
    return SP.reshape(Y,(n,t),order='F')

def fast_dot(A, B):
    f_dot = SP.linalg.get_blas_funcs("gemm", arrays=(A.T, B.T))
    return f_dot(alpha=1.0, a = A.T, b = B.T, trans_a = True, trans_b = True)
    
def fast_kron(A, B):
    f_kron = A[:,SP.newaxis,:,SP.newaxis] * B[SP.newaxis,:, SP.newaxis,:]
    f_kron = SP.reshape(f_kron, [A.shape[0] * B.shape[0], A.shape[1] * B.shape[1]])
    return f_kron


def compute_r2(Y1, Y2):
    """
    return list of squared correlation coefficients (one per task)
    """
    if Y1.ndim==1:
        Y1 = SP.reshape(Y1,(Y1.shape[0],1))
    if Y2.ndim==1:
        Y2 = SP.reshape(Y2,(Y2.shape[0],1))

    t = Y1.shape[1]
    r2 = []
    for i in range(t):
        _r2 = SP.corrcoef(Y1[:,i],Y2[:,i])[0,1]**2
        r2.append(_r2)
    r2 = SP.array(r2)
    return r2
 
    
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s \r' % (prefix, bar, percent, suffix)) #, end = '\r'
    # Print New Line on Complete
    if iteration == total: 
        print()
