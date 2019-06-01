# -*- coding: utf-8 -*-
"""
@author: seykia
"""

import numpy as np
import time
from core.util.utilities import read_phenomics_data
import core.util.initialize as initialize
import core.covariance.linear as linear
import core.covariance.se as se
import core.covariance.composite as composite
from core.covariance.diag import DiagIsoCF
import core.gp.gp_sMTGPTR as gp_sMTGPTR
import core.optimize.optimize_tensor as optimize_tensor
from sklearn.preprocessing import StandardScaler, Imputer
import tensorly as tl
from tensorly.base import partial_unfold, partial_fold, fold
from core.util.normod import extreme_value_prob, normative_prob_map, extreme_value_prob_fit
from sklearn.linear_model import LinearRegression
import core.gp.gp_base as gp_base
import core.likelihood.likelihood_base as lik
import core.optimize.optimize_base as optimize_base
from sklearn.metrics import roc_auc_score
import nibabel as nib
from scipy.io import savemat
from functools import reduce
tl.set_backend('numpy') 

############################# PRARAMTERs and PATHs ############################
runs = 20
fixed_effect = True

method = 'sMT_GPTR' # 'sMT_GPTR' for scalable multi-task Gaussian process regression and 'ST' for single-task experiments
b = 10 # the number of signal components in tucker decompositions, 3, 5, 10, 15, ...
nb = 5 # the number of noise components in tucker decompositions, 1, 3, 5, 10, ...

main_dir = 'PATH to PHENOMICS DATA/ds000030_R1.0.5/'
save_path = 'PATH TO SAVE THE RESULTS'

###################### Loading Training Data ##################################
phenotypes = ['health']
# Cropping box
x_from = 8; x_to = 57; y_from = 8; y_to = 69; z_from = 1; z_to = 41; 

control_fmri_data, control_phenotype_data = read_phenomics_data(main_dir, 
                diagnosis='CONTROL', task_name = 'taskswitch' , cope_num = 1, phenotypes = phenotypes)
SCHZ_fmri_data, SCHZ_phenotype_data = read_phenomics_data(main_dir, 
                diagnosis='SCHZ', task_name = 'taskswitch' , cope_num = 1, phenotypes = phenotypes)
ADHD_fmri_data, ADHD_phenotype_data = read_phenomics_data(main_dir, 
                diagnosis='ADHD', task_name = 'taskswitch' , cope_num = 1, phenotypes = phenotypes)
BIPL_fmri_data, BIPL_phenotype_data = read_phenomics_data(main_dir, 
                diagnosis='BIPOLAR', task_name = 'taskswitch' , cope_num = 1, phenotypes = phenotypes)

original_image_size = list(control_fmri_data.shape)

control_phenotype_data[control_phenotype_data<-100] = np.nan
SCHZ_phenotype_data[SCHZ_phenotype_data<-100] = np.nan
ADHD_phenotype_data[ADHD_phenotype_data<-100] = np.nan
BIPL_phenotype_data[BIPL_phenotype_data<-100] = np.nan

control_fmri_data = control_fmri_data[:,x_from:x_to,y_from:y_to,z_from:z_to]
SCHZ_fmri_data = SCHZ_fmri_data[:,x_from:x_to,y_from:y_to,z_from:z_to]
ADHD_fmri_data = ADHD_fmri_data[:,x_from:x_to,y_from:y_to,z_from:z_to]
BIPL_fmri_data = BIPL_fmri_data[:,x_from:x_to,y_from:y_to,z_from:z_to]

Y_shape = control_fmri_data.shape[1:]

voxel_num = np.prod(control_fmri_data.shape[1:])

if (method == 'ST'):
    control_fmri_data = np.reshape(control_fmri_data, [control_fmri_data.shape[0],voxel_num])
    SCHZ_fmri_data = np.reshape(SCHZ_fmri_data, [SCHZ_fmri_data.shape[0],voxel_num])
    ADHD_fmri_data = np.reshape(ADHD_fmri_data, [ADHD_fmri_data.shape[0],voxel_num])
    BIPL_fmri_data = np.reshape(BIPL_fmri_data, [BIPL_fmri_data.shape[0],voxel_num])
    
SCHZ_sample_num = SCHZ_fmri_data.shape[0]
ADHD_sample_num = ADHD_fmri_data.shape[0]
BIPL_sample_num = BIPL_fmri_data.shape[0]
patient_sample_num = SCHZ_sample_num + ADHD_sample_num + BIPL_sample_num
control_sample_num = control_fmri_data.shape[0]

################################ Training Tensor-GP ###########################

train_num = int(np.floor(control_sample_num / 3 ))
labels = np.zeros([patient_sample_num + control_sample_num - 2*train_num,])
labels[control_sample_num - 2*train_num:,] = 1 
diagnosis_labels = np.zeros_like(labels)
diagnosis_labels[control_sample_num - train_num:control_sample_num - train_num+SCHZ_sample_num] = 1
diagnosis_labels[control_sample_num - train_num + SCHZ_sample_num-5:control_sample_num - 
                 train_num + SCHZ_sample_num + ADHD_sample_num] = 2
diagnosis_labels[control_sample_num - train_num + SCHZ_sample_num + ADHD_sample_num:] = 3 

elapsed_time_opt = np.zeros([runs,])
elapsed_time_est = np.zeros([runs,])
r2 = np.zeros([runs,])
r2_2 = np.zeros([3,runs])
auc_all = np.zeros([runs,])
auc_SCHZ = np.zeros([runs,])
auc_ADHD = np.zeros([runs,])
auc_BIPL = np.zeros([runs,])
evd_shape = np.zeros([runs,])

for r in range(runs):
    rand_idx = np.random.permutation(control_sample_num)
    train_idx = rand_idx[0:train_num]
    test_idx = np.setdiff1d(np.array(range(control_sample_num)),train_idx)
    
    X_train = control_phenotype_data[train_idx,]
    X_test = np.concatenate((control_phenotype_data[test_idx,:], SCHZ_phenotype_data, 
                             ADHD_phenotype_data, BIPL_phenotype_data))
        
    Y_train = control_fmri_data[train_idx,:]
    Y_test = np.concatenate((control_fmri_data[test_idx,:], SCHZ_fmri_data,
                             ADHD_fmri_data, BIPL_fmri_data))
    
    X_imputer = Imputer()
    X_imputer.fit(X_train)
    X_train = X_imputer.transform(X_train)
    X_test = X_imputer.transform(X_test)
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    
    Y_scaler = StandardScaler()
    Y_train = Y_scaler.fit_transform(partial_unfold(Y_train,mode=0,skip_begin=0))
    Y_test = partial_unfold(Y_test,mode=0,skip_begin=0)
    
    if fixed_effect:        
        temp = np.zeros(Y_train.shape,dtype=np.float32)    
        regresors = []
        for i in range(Y_train.shape[1]):
            reg = LinearRegression() 
            reg.fit(X_train,Y_train[:,i])
            temp[:,i] = reg.predict(X_train)
            regresors.append(reg)
            if (i + 1) % 1000 == 0:
                print(i+1)
        Y_train = Y_scaler.inverse_transform(Y_train) - Y_scaler.inverse_transform(temp) 
        del temp
        Y_scaler2 = StandardScaler()
        Y_train = Y_scaler2.fit_transform(Y_train)
    else:
        Y_scaler2 = Y_scaler
    
    if (method == 'sMT_GPTR'):
        basis_num = [b,b,b]
        noise_basis_num = [nb,nb,nb]
        Y_train = partial_fold(Y_train, mode=0, shape = [Y_train.shape[0],
                            control_fmri_data.shape[1],control_fmri_data.shape[2],control_fmri_data.shape[3]])
        
        hyperparams, Ifilter, bounds = initialize.init('Zero', Y_train, X_train,{})
        covar_c = list()
        covar_s = list()
        covar_r = list()
        covar_o = list()
        for i in range(Y_train.ndim-1):
            n_dim = Y_train.shape[0] * np.prod(basis_num) / basis_num[i]
            covar_c.append(composite.SumCF(n_dimensions = n_dim))
            covar_c[i].append_covar(linear.LinearCF(n_dimensions = n_dim))
            covar_c[i].append_covar(se.SqExpCF(n_dimensions = n_dim))
            covar_c[i].append_covar(DiagIsoCF(n_dimensions = n_dim))
            n_dim = Y_train.shape[0] * np.prod(noise_basis_num) / noise_basis_num[i]
            covar_s.append(composite.SumCF(n_dimensions = n_dim))
            covar_s[i].append_covar(linear.LinearCF(n_dimensions = n_dim))
            covar_s[i].append_covar(se.SqExpCF(n_dimensions = n_dim))
            covar_s[i].append_covar(DiagIsoCF(n_dimensions = n_dim))
            
        covar_r.append(composite.SumCF(n_dimensions = X_train.shape[1]))
        covar_r[0].append_covar(linear.LinearCF(n_dimensions = X_train.shape[1]))
        covar_r[0].append_covar(se.SqExpCF(n_dimensions = X_train.shape[1]))
        covar_r[0].append_covar(DiagIsoCF(n_dimensions = X_train.shape[1]))
        covar_r[0].X = X_train    
        covar_o.append(DiagIsoCF(n_dimensions = X_train.shape[1]))
        covar_o[0].X = X_train
        covar_o[0]._K = np.eye(X_train.shape[0])
        gp = gp_sMTGPTR.sMTGPTR(basis_num = basis_num, noise_basis_num = noise_basis_num, 
                                covar_c = covar_c, covar_r = covar_r, covar_s = covar_s, covar_o = covar_o)
        gp.setData(Y = Y_train)
        
        t_start = time.clock()
        hyperparams_opt,lml_opt = optimize_tensor.opt_hyper(gp, hyperparams, bounds = bounds, Ifilter = Ifilter)
        t_stop = time.clock()
        elapsed_time_opt[r,] = t_stop - t_start
    
    elif method == 'ST':
        opts = {'messages':False}
        n_task = Y_test.shape[1]
        time_opt = np.zeros([n_task,])
        time_est = np.zeros([n_task,])
        hp = []
        for i in range(n_task):
            hyperparams, Ifilter, bounds = initialize.init('GPbase_nonLIN', Y_train[:,i].T, X_train, None)
            covariance = composite.SumCF(n_dimensions = X_train.shape[1])
            covariance.append_covar(linear.LinearCF(n_dimensions = X_train.shape[1]))
            covariance.append_covar(se.SqExpCF(n_dimensions = X_train.shape[1]))
            covariance.append_covar(DiagIsoCF(n_dimensions = X_train.shape[1]))
            likelihood = lik.GaussIsoLik()
            gp = gp_base.GP(covar=covariance, likelihood=likelihood)
            gp.setData(Y = Y_train[:,i:i+1], X_r = X_train) 
            t_start = time.clock()
            hyperparams_opt,lml_opt = optimize_base.opt_hyper(gp, hyperparams, bounds = bounds, 
                                                              Ifilter = Ifilter, opts=opts)
            t_stop = time.clock()
            time_opt[i] = t_stop - t_start
            hp.append(hyperparams_opt)
            if (i + 1) % 1000 == 0:
                print(i+1) 
        elapsed_time_opt[r,] = np.sum(time_opt)
        
    #################################### Predicting ###########################
        
    if method == 'ST':
        Y_pred_MT = np.zeros(np.shape(Y_test))
        Y_pred_cov_MT = np.zeros(np.shape(Y_test))
        s_n2_MT = np.zeros(Y_test.shape[1])
        for i in range(n_task):
            t_start = time.clock()
            Y_pred_MT[:,i], Y_pred_cov_MT[:,i]  = gp.predict(hp[i], Xstar_r = X_test)
            t_stop = time.clock()
            time_est[i] = t_stop - t_start
            s_n2_MT[i] = np.exp(2 * hyperparams_opt['lik'])
            if (i + 1) % 1000 == 0:
                print(i+1)  
        elapsed_time_est[r,] = np.sum(time_est)
        Y_pred_MT = Y_scaler2.inverse_transform(Y_pred_MT)
        Y_pred_cov_MT = Y_pred_cov_MT * Y_scaler2.var_
        s_n2_MT = s_n2_MT * Y_scaler2.var_
    elif method == 'sMT_GPTR':
        t_start = time.clock()
        Y_pred_MT, Y_pred_cov_MT = gp.predict(hyperparams_opt, Xstar_r = X_test, compute_cov = True)
        t_stop = time.clock()
        elapsed_time_est[r,] = t_stop - t_start
        Y_pred_MT = Y_scaler2.inverse_transform(Y_pred_MT)
        Y_pred_cov_MT = Y_pred_cov_MT * Y_scaler2.var_
        s_n2_MT = reduce(np.kron,[np.diag(reduce(np.dot, [gp.nbasis[0], 
                         covar_s[0].K(hyperparams_opt['covar_s'][0]), gp.nbasis[0].T])),
                         np.diag(reduce(np.dot, [gp.nbasis[1], covar_s[1].K(hyperparams_opt['covar_s'][1]), gp.nbasis[1].T])),
                         np.diag(reduce(np.dot, [gp.nbasis[2], covar_s[2].K(hyperparams_opt['covar_s'][2]), gp.nbasis[2].T]))]) * Y_scaler2.var_    
    if fixed_effect:
        Y_pred_linear = np.zeros(np.shape(Y_test))
        for i in range(Y_test.shape[1]):
            Y_pred_linear[:,i] = regresors[i].predict(X_test)      
        Y_pred_linear = Y_scaler.inverse_transform(Y_pred_linear)
        Y_pred_MT = Y_pred_MT + Y_pred_linear
    
    ####################################### Evaluation ########################
                                        
    NPMs_MT = normative_prob_map(Y_test, Y_pred_MT, Y_pred_cov_MT, s_n2_MT)
    NPMs_MT[~np.isfinite(NPMs_MT)] = 0
    EVD_params = extreme_value_prob_fit(NPMs_MT[:train_num,:], 0.01)
    abnormal_probs_MT = extreme_value_prob(EVD_params, NPMs_MT[train_num:,:], 0.01)
    auc_all[r,] = roc_auc_score(labels, abnormal_probs_MT)   
    auc_SCHZ[r] = roc_auc_score(labels[(diagnosis_labels==0) | (diagnosis_labels==1),], abnormal_probs_MT[(diagnosis_labels==0) | (diagnosis_labels==1),])
    auc_ADHD[r] = roc_auc_score(labels[(diagnosis_labels==0) | (diagnosis_labels==2),], abnormal_probs_MT[(diagnosis_labels==0) | (diagnosis_labels==2),])
    auc_BIPL[r] = roc_auc_score(labels[(diagnosis_labels==0) | (diagnosis_labels==3),], abnormal_probs_MT[(diagnosis_labels==0) | (diagnosis_labels==3),])    
         
    print ('Results'  + 'AUC = %f' %(auc_all[r,]))
    
    ################################## Saving Results ######################### 
    
    ex_img = nib.load(main_dir + 'derivatives/task/sub-10159/taskswitch.feat/example_func.nii.gz')
    NPMs_MT = fold(NPMs_MT, mode=0, shape = [NPMs_MT.shape[0],Y_shape[0],Y_shape[1],Y_shape[2]])
    original_image_size[0] = NPMs_MT.shape[0]
    temp = np.zeros(original_image_size, dtype=np.float32)
    temp[:,x_from:x_to,y_from:y_to,z_from:z_to] = NPMs_MT
    image = nib.Nifti1Image(np.transpose(temp, [1,2,3,0]), ex_img.affine, ex_img.header)
    nib.save(image, save_path + method + '_' + str(b) + '_' + str(nb) + '_NPMs_' + str(r) + '.nii.gz')
    
    savemat(save_path + method + '_' + str(b) + '_' + str(nb) + '_results.mat',
            {'elapsed_time_opt': elapsed_time_opt, 'elapsed_time_est': elapsed_time_est, 
             'auc_all':auc_all, 'auc_SCHZ':auc_SCHZ, 'auc_ADHD':auc_ADHD, 'auc_BIPL':auc_BIPL,
             'hyperparams':hyperparams_opt})
