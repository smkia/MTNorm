# -*- coding: utf-8 -*-
"""
This script regenerates the experimental results.

Created on Mon Jan 22 13:27:16 2018

@author: Seyed Mostafa Kia (m.kia83@gmail.com)
"""

import numpy as np
from nilearn.input_data import MultiNiftiMasker
from nilearn.datasets import fetch_miyawaki2008
from sklearn.preprocessing import StandardScaler
import core.util.initialize as initialize
import core.covariance.linear as linear
import core.covariance.se as se
import core.covariance.composite as composite
from core.covariance.diag import DiagIsoCF
import core.likelihood.likelihood_base as lik
import core.gp.gp_base as gp_base
import core.optimize.optimize_base as optimize_base
from scipy.io import savemat
from core.util.utilities import compute_r2
from sklearn.decomposition import PCA
import core.gp.gp_sMTGPR as sMTGPR
import core.gp.gp_kronprod as gp_kronprod
from core.util.normod import extreme_value_prob, normative_prob_map, extreme_value_prob_fit
from sklearn.metrics import roc_auc_score
import time

###############################################################################

method = 'STGPR' # Select the method among MT_Kronprod, sMT_GPR, STGPR 
save_path = 'path for saving the results'
runs = 10 # Select the number of runs

############################ Loading Data #####################################
miyawaki_dataset = fetch_miyawaki2008()

Y_random_filenames = miyawaki_dataset.func[12:]
Y_figure_filenames = miyawaki_dataset.func[:12]
X_random_filenames = miyawaki_dataset.label[12:]
X_figure_filenames = miyawaki_dataset.label[:12]
X_shape = (10, 10)

masker = MultiNiftiMasker(mask_img=miyawaki_dataset.mask, detrend=True,
                          standardize=False)
masker.fit()
fmri_random = masker.transform(Y_random_filenames)
fmri_figure = masker.transform(Y_figure_filenames)

pattern_random = [] 
for x in X_random_filenames:
    pattern_random.append(np.reshape(np.loadtxt(x, dtype=np.int, delimiter=','),
                              (-1,) + X_shape, order='F'))

pattern_figure= []
for x in X_figure_filenames:
    pattern_figure.append(np.reshape(np.loadtxt(x, dtype=np.int, delimiter=','),
                             (-1,) + X_shape, order='F'))
                             
fmri_random = np.vstack([y[2:] for y in fmri_random])
pattern_random = np.vstack([x[:-2] for x in pattern_random]).astype(float)
fmri_figure = np.vstack([y[2:] for y in fmri_figure])
pattern_figure = np.vstack([x[:-2] for x in pattern_figure]).astype(float)

pattern_random = np.reshape(pattern_random,(pattern_random.shape[0],pattern_random.shape[1] * pattern_random.shape[2]))
pattern_figure = np.reshape(pattern_figure,(pattern_figure.shape[0],pattern_figure.shape[1] * pattern_figure.shape[2]))

n_pixels = pattern_random.shape[1] 
n_task = fmri_random.shape[1]

fmri_random = np.float64(fmri_random[pattern_random[:, 0] != -1])
pattern_random = np.float64(pattern_random[pattern_random[:, 0] != -1])
fmri_figure = np.float64(fmri_figure[pattern_figure[:, 0] != -1])
pattern_figure = np.float64(pattern_figure[pattern_figure[:, 0] != -1])
n = pattern_figure.shape[0]

###############################################################################
###############################################################################
opts = {'messages':False}
if method == 'sMT_GPR':
    basis_num = [10,25,50,100,250,500,1000]
    elapsed_time_opt = np.zeros([runs,len(basis_num)])
    elapsed_time_est = np.zeros([runs,len(basis_num)])
    r2_MT = np.zeros([runs,len(basis_num)])
    auc_MT = np.zeros([runs,len(basis_num)])
    for r in range(runs):
        train_idx = np.random.permutation(pattern_random.shape[0])
        train_idx = train_idx[0:pattern_random.shape[0]-n]
        test_idx = np.setdiff1d(np.array(range(pattern_random.shape[0])),train_idx)

        X_train = pattern_random[train_idx,:]
        X_test = np.concatenate((pattern_random[test_idx,:],pattern_figure))
        Y_train = fmri_random[train_idx,:]
        Y_test = np.concatenate((fmri_random[test_idx,:],fmri_figure))
        labels = np.zeros([Y_test.shape[0],1])
        labels[n:] = 1
    
        X_scaler = StandardScaler()
        X_scaler.fit(X_train)
        X_train = X_scaler.transform(X_train)
        X_test = X_scaler.transform(X_test)
    
        Y_scaler = StandardScaler()
        Y_scaler.fit(Y_train)
        Y_train = Y_scaler.transform(Y_train)
        
        for b in range(len(basis_num)):
            pca = PCA(n_components = basis_num[b])
            Y_train_hat = pca.fit_transform(Y_train)
            B = pca.components_.T
             
            hyperparams, Ifilter, bounds = initialize.init('MTGPR', Y_train_hat.T, X_train,{})
            covar_c = composite.SumCF(n_dimensions = Y_train_hat.shape[0])
            covar_c.append_covar(linear.LinearCF(n_dimensions = Y_train_hat.shape[0]))
            covar_c.append_covar(se.SqExpCF(n_dimensions = Y_train_hat.shape[0]))
            covar_c.append_covar(DiagIsoCF(n_dimensions = Y_train_hat.shape[0]))
            covar_c.X = Y_train_hat.T
            
            covar_r = composite.SumCF(n_dimensions = X_train.shape[1])
            covar_r.append_covar(linear.LinearCF(n_dimensions = X_train.shape[1]))
            covar_r.append_covar(se.SqExpCF(n_dimensions = X_train.shape[1]))
            covar_r.append_covar(DiagIsoCF(n_dimensions = X_train.shape[1]))
            covar_r.X = X_train
            
            likelihood = lik.GaussIsoLik()
            
            gp = sMTGPR.sMTGPR(covar_r = covar_r, covar_c = covar_c, likelihood = likelihood, basis = B)
            gp.setData(Y = Y_train, Y_hat = Y_train_hat, X = X_train)
            
            # Training: optimize hyperparameters
            t_start = time.clock()
            hyperparams_opt,lml_opt = optimize_base.opt_hyper(gp, hyperparams, bounds = bounds, Ifilter = Ifilter)
            t_stop = time.clock()
            elapsed_time_opt[r,b] = t_stop - t_start
            
            # Testing
            t_start = time.clock()
            Y_pred_MT, Y_pred_cov_MT = gp.predict(hyperparams_opt, Xstar_r = X_test, compute_cov = True)
            t_stop = time.clock()
            elapsed_time_est[r,b] = t_stop - t_start
            
            Y_pred_MT = Y_scaler.inverse_transform(Y_pred_MT)
            Y_pred_cov_MT = Y_pred_cov_MT * Y_scaler.var_
            s_n2_MT = likelihood.Kdiag(hyperparams_opt['lik'], n_task) * Y_scaler.var_
            
            r2_MT[r,b] = np.mean(compute_r2(Y_test, Y_pred_MT))
            NPMs_MT = normative_prob_map(Y_test, Y_pred_MT, Y_pred_cov_MT, s_n2_MT)
            EVD_params = extreme_value_prob_fit(NPMs_MT[np.squeeze(labels==0),:], 0.05)
            abnormal_probs_MT = extreme_value_prob(EVD_params, NPMs_MT, 0.05)
            auc_MT[r,b] = roc_auc_score(labels, abnormal_probs_MT)    
            
            print ('sMT-GPR_' + str(basis_num[b]) + ': R2 = %f, AUC = %f' %(r2_MT[r,b], auc_MT[r,b]))
            
            savemat(save_path + 'results_miyawaki_normative_sMTGPR.mat',{'r2_MT': r2_MT,
                                'elapsed_time_opt': elapsed_time_opt, 'elapsed_time_est': elapsed_time_est, 
                                                          'auc_MT':auc_MT})
elif method == 'STGPR':
    elapsed_time_opt = np.zeros([runs, n_task])
    elapsed_time_est = np.zeros([runs, n_task])
    r2_ST = np.zeros([runs,n_task])
    auc_ST = np.zeros([runs,])
    for r in range(runs):
        train_idx = np.random.permutation(pattern_random.shape[0])
        train_idx = train_idx[0:pattern_random.shape[0]-n]
        test_idx = np.setdiff1d(np.array(range(pattern_random.shape[0])),train_idx)

        X_train = pattern_random[train_idx,:]
        X_test = np.concatenate((pattern_random[test_idx,:],pattern_figure))
        Y_train = fmri_random[train_idx,:]
        Y_test = np.concatenate((fmri_random[test_idx,:],fmri_figure))
        labels = np.zeros([Y_test.shape[0],1])
        labels[n:] = 1
    
        X_scaler = StandardScaler()
        X_scaler.fit(X_train)
        X_train = X_scaler.transform(X_train)
        X_test = X_scaler.transform(X_test)
    
        Y_scaler = StandardScaler()
        Y_scaler.fit(Y_train)
        Y_train = Y_scaler.transform(Y_train)
        
        Y_pred_base = np.zeros(np.shape(Y_test))
        Y_pred_cov_base = np.zeros(np.shape(Y_test))
        s_n2_base = np.zeros(Y_test.shape[1])
        
        hp = []
        for i in range(n_task):
            hyperparams, Ifilter, bounds = initialize.init('STGPR', Y_train[:,i].T, X_train, None)
            covariance = composite.SumCF(n_dimensions = X_train.shape[1])
            covariance.append_covar(linear.LinearCF(n_dimensions = X_train.shape[1]))
            covariance.append_covar(se.SqExpCF(n_dimensions = X_train.shape[1]))
            likelihood = lik.GaussIsoLik()
            gp = gp_base.GP(covar=covariance, likelihood=likelihood)
            gp.setData(Y = Y_train[:,i:i+1], X_r = X_train)
            t_start = time.clock()
            hyperparams_opt,lml_opt = optimize_base.opt_hyper(gp, hyperparams, bounds = bounds, Ifilter = Ifilter, opts=opts)
            t_stop = time.clock()
            elapsed_time_opt[r,i] = t_stop - t_start
            hp.append(hyperparams_opt)
            t_start = time.clock()
            Y_pred_base[:,i], Y_pred_cov_base[:,i] = gp.predict(hyperparams_opt, Xstar_r = X_test)
            t_stop = time.clock()
            elapsed_time_est[r,i] = t_stop - t_start
            s_n2_base[i] = np.exp(2 * hyperparams_opt['lik'])
            if np.mod(i,100) == 0:
                print(i)
                
        Y_pred_base = Y_scaler.inverse_transform(Y_pred_base)
        Y_pred_cov_base = Y_pred_cov_base * Y_scaler.var_
        s_n2_base = s_n2_base * Y_scaler.var_
        
        r2_ST[r,:] = compute_r2(Y_test, Y_pred_base)
        NPMs_ST = normative_prob_map(Y_test, Y_pred_base, Y_pred_cov_base, s_n2_base)
        EVD_params = extreme_value_prob_fit(NPMs_ST[np.squeeze(labels==0),:], 0.05)
        abnormal_probs_ST = extreme_value_prob(EVD_params,NPMs_ST, 0.05)
        auc_ST[r,] = roc_auc_score(labels, abnormal_probs_ST)
        
        print ('STGPR_' + str(r) + ': R2 = %f, AUC = %f' %(np.mean(r2_ST[r,:]), auc_ST[r,]))
                                                      
        savemat(save_path + 'results_miyawaki_normative_STGPR.mat',{'r2_ST': r2_ST, 
                            'elapsed_time_opt': elapsed_time_opt, 'elapsed_time_est': elapsed_time_est,
                                                          'auc_ST':auc_ST})
                                                          
elif method == 'MT_Kronprod':
    elapsed_time_opt = np.zeros([runs,])
    elapsed_time_est = np.zeros([runs,])
    r2_MT = np.zeros([runs,])
    auc_MT = np.zeros([runs,])
    for r in range(runs):
        train_idx = np.random.permutation(pattern_random.shape[0])
        train_idx = train_idx[0:pattern_random.shape[0]-n]
        test_idx = np.setdiff1d(np.array(range(pattern_random.shape[0])),train_idx)

        X_train = pattern_random[train_idx,:]
        X_test = np.concatenate((pattern_random[test_idx,:],pattern_figure))
        Y_train = fmri_random[train_idx,:]
        Y_test = np.concatenate((fmri_random[test_idx,:],fmri_figure))
        labels = np.zeros([Y_test.shape[0],1])
        labels[n:] = 1
    
        X_scaler = StandardScaler()
        X_scaler.fit(X_train)
        X_train = X_scaler.transform(X_train)
        X_test = X_scaler.transform(X_test)
    
        Y_scaler = StandardScaler()
        Y_scaler.fit(Y_train)
        Y_train = Y_scaler.transform(Y_train)
        
        hyperparams, Ifilter, bounds = initialize.init('MTGPR', Y_train.T, X_train,{})
        covar_c = composite.SumCF(n_dimensions = Y_train.shape[0])
        covar_c.append_covar(linear.LinearCF(n_dimensions = Y_train.shape[0]))
        covar_c.append_covar(se.SqExpCF(n_dimensions = Y_train.shape[0]))
        covar_c.append_covar(DiagIsoCF(n_dimensions = Y_train.shape[0]))
        covar_c.X = Y_train.T
        covar_r = composite.SumCF(n_dimensions = X_train.shape[1])
        covar_r.append_covar(linear.LinearCF(n_dimensions = X_train.shape[1]))
        covar_r.append_covar(se.SqExpCF(n_dimensions = X_train.shape[1]))
        covar_r.append_covar(DiagIsoCF(n_dimensions = X_train.shape[1]))
        likelihood = lik.GaussIsoLik()
        covar_r.X = X_train
        
        gp = gp_kronprod.KronProdGP(covar_r = covar_r, covar_c = covar_c, likelihood = likelihood)
        gp.setData(Y = Y_train, X = X_train)
        
        # Training: optimize hyperparameters
        t_start = time.clock()
        hyperparams_opt,lml_opt = optimize_base.opt_hyper(gp, hyperparams, bounds = bounds, Ifilter = Ifilter)
        t_stop = time.clock()
        elapsed_time_opt[r,] = t_stop - t_start
        
        # Testing
        t_start = time.clock()
        Y_pred_MT, Y_pred_cov_MT = gp.predict(hyperparams_opt, Xstar_r = X_test, compute_cov = True)
        t_stop = time.clock()
        elapsed_time_est[r,] = t_stop - t_start
        
        Y_pred_MT = Y_scaler.inverse_transform(Y_pred_MT)
        Y_pred_cov_MT = Y_pred_cov_MT * Y_scaler.var_
        s_n2_MT = likelihood.Kdiag(hyperparams_opt['lik'], n_task) * Y_scaler.var_
        
        r2_MT[r,] = np.mean(compute_r2(Y_test, Y_pred_MT))
        NPMs_MT = normative_prob_map(Y_test, Y_pred_MT, Y_pred_cov_MT, s_n2_MT)
        EVD_params = extreme_value_prob_fit(NPMs_MT[np.squeeze(labels==0),:], 0.05)
        abnormal_probs_MT = extreme_value_prob(EVD_params, NPMs_MT, 0.05)
        auc_MT[r,] = roc_auc_score(labels, abnormal_probs_MT)    
        
        print ('MT-Kronprod' + ': R2 = %f, AUC = %f' %(r2_MT[r,], auc_MT[r,]))
        
        savemat(save_path + 'results_miyawaki_normative_Kronprod.mat',{'r2_MT': r2_MT, 
                'elapsed_time_opt': elapsed_time_opt, 'elapsed_time_est': elapsed_time_est, 
                                                      'auc_MT':auc_MT})
                                                      
###############################################################################
###############################################################################