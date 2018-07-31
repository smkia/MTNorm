import pdb
import logging as LG
import scipy as SP
import scipy.linalg as LA
import copy
from gpbase import GPBASE
from core.util.utilities import ravel,unravel,printProgressBar
import tensorly as tl
from tensorly.decomposition import partial_tucker
from tensorly.base import unfold

class sMTGPTR(GPBASE):

    __slots__ = ['covar_c','covar_r','covar_o','covar_s','bn', 'nbn','out_dims','in_dims']
    
    def __init__(self, basis_num = [1,1,1], noise_basis_num = [1,1,1], covar_r=None,covar_c=None,covar_o=None,covar_s=None,prior=None):
        self.covar_r = covar_r
        self.covar_c = covar_c
        self.covar_s = covar_s
        self.covar_o = covar_o
        assert len(covar_c)==len(covar_s), 'Signal and noise covariance matrix have different dimensions!!!'
        self.bn = basis_num
        self.nbn = noise_basis_num
        self.out_dims = len(covar_c)
        self.in_dims = len(covar_r)
        self.likelihood  = None
        self._covar_cache = None
        self.prior = None

    def setPrior(self,prior):
        self.prior = None

    def setData(self,Y=None,X=None,X_r=None,X_o=None,**kwargs):
        """
        set data
        Y:    Outputs [n x t_1 x t_2 x ... x t_c]
        """
        self.Y = Y
        self.n = Y.shape[0]
        self.t = SP.zeros([Y.ndim-1,],dtype=int)
        for i in range(1,Y.ndim):
            self.t[i-1] = Y.shape[i]
        self.nt = SP.prod(Y.shape)
        if X_r!=None:
            X = X_r
        if X!=None:
            self.covar_r.X = X
        if X_o!=None:
            self.covar_o.X = X_o
            
        self.Y_hat, self.basis = partial_tucker(Y, modes = range(1,Y.ndim), ranks=self.bn, init='svd', tol=10e-5)
        for i in range(Y.ndim-1):
            self.covar_c[i].X = unfold(self.Y_hat, mode = i+1)
        reconstruction = SP.zeros(self.Y.shape)
        for i in range(self.Y_hat.shape[0]):
            reconstruction[i,:] = tl.tucker_to_tensor(self.Y_hat[i,:], self.basis)        
        res = Y - reconstruction
        temp, self.nbasis = partial_tucker(res, modes = range(1,Y.ndim), ranks=self.nbn, init='svd', tol=10e-5)
        for i in range(Y.ndim-1):
            self.covar_s[i].X = unfold(temp, mode = i+1)
        
        self._invalidate_cache()


    def predict(self,hyperparams, Xstar_r, compute_cov = False):
        """
        predict over new training points
        """
        KV = self.get_covariances(hyperparams)
        self.covar_r[0].Xcross = Xstar_r
        Kstar_r = self.covar_r[0].Kcross(hyperparams['covar_r'][0])
        USUr = SP.dot(SP.sqrt(1./KV['S_o'][0]) * KV['U_o'][0],KV['Utilde_r'][0])
        S = KV['Stilde_rc']
        RUSUrYhat = SP.dot(Kstar_r.T,SP.dot(USUr,unravel(ravel(KV['UYtildeU_rc']) * 1./S, self.n, SP.prod(self.nbn))))
        usuc = list()
        for i in range(self.out_dims):
            usuc.append(SP.dot(self.basis[i], SP.dot(KV['K_c'][i], SP.dot(self.basis[i].T, 
                               SP.dot(self.nbasis[i], SP.dot(KV['USi_s'][i].T,KV['Utilde_c'][i]))))).T) 
        if SP.prod(self.t) <= 10000 :
            USUC = reduce(SP.kron, usuc)
            Ystar = SP.dot(RUSUrYhat,USUC)
        else:
            Ystar = SP.zeros([Xstar_r.shape[0],SP.prod(self.t)])
            if self.out_dims == 1:
                for j in range(self.t[0]):
                    USUC = usuc[0][:,j]
                    Ystar[:,j] = SP.dot(RUSUrYhat,USUC)
            elif self.out_dims == 2:
                for j in range(self.t[0]):
                    for k in range(self.t[1]):
                        USUC = reduce(SP.kron, [usuc[0][:,j],usuc[1][:,k]])
                        Ystar[:,k + j*self.t[1]] = SP.dot(RUSUrYhat,USUC)
            elif self.out_dims == 3: 
               for j in range(self.t[0]):
                    for k in range(self.t[1]):
                        for l in range(self.t[2]):
                            USUC = reduce(SP.kron, [usuc[0][:,j],usuc[1][:,k],usuc[2][:,l]])
                            Ystar[:,l + k*self.t[2] + j*(self.t[1]*self.t[2])] = SP.dot(RUSUrYhat,USUC)
        Ystar_covar = []    
        if compute_cov:
            if self.nt < 10000:
                B = reduce(SP.kron, self.basis)
                C = SP.dot(B, SP.dot(reduce(SP.kron,KV['K_c']), B.T))
                USUC = reduce(SP.kron, usuc)
                R_star_star = self.covar_r[0].K(hyperparams['covar_r'][0], Xstar_r)         
                temp = SP.kron(USUC.T,SP.dot(Kstar_r.T,USUr)) 
                Ystar_covar = SP.kron(SP.diag(C), SP.diag(R_star_star)) - SP.sum((1./S * temp).T * temp.T, axis = 0)
                Ystar_covar = unravel(Ystar_covar, Xstar_r.shape[0], SP.prod(self.t))
            elif SP.prod(self.t) < 10000:
                Ystar_covar = SP.zeros([Xstar_r.shape[0], SP.prod(self.t)])
                B = reduce(SP.kron, self.basis)
                C = SP.dot(B, SP.dot(reduce(SP.kron,KV['K_c']), B.T))
                USUC = reduce(SP.kron, usuc)
                printProgressBar(0, Xstar_r.shape[0], prefix = 'Computing perdiction varaince:', suffix = 'Complete', length = 20)            
                for i in range(Xstar_r.shape[0]):
                    R_star_star = self.covar_r[0].K(hyperparams['covar_r'][0], SP.expand_dims(Xstar_r[i,:],axis=0))
                    self.covar_r[0].Xcross = SP.expand_dims(Xstar_r[i,:],axis=0)
                    R_tr_star = self.covar_r[0].Kcross(hyperparams['covar_r'][0])
                    r = SP.dot(R_tr_star.T, USUr)
                    q = SP.diag(SP.kron(C, R_star_star))
                    t = SP.zeros([SP.prod(self.t)])
                    for j in range(SP.prod(self.t)):
                        temp = SP.kron(USUC[:,j], r) 
                        t[j,] = SP.sum((1./S * temp).T * temp.T, axis = 0)
                    Ystar_covar[i,:] = q - t
                    if (i + 1) % 50 == 0:
                        printProgressBar(i+1, Xstar_r.shape[0], prefix = 'Computing perdiction varaince:', suffix = 'Complete', length = 20)
                self.covar_r[0].Xcross = Xstar_r 
            elif Xstar_r.shape[0] < 2000:
                Ystar_covar = SP.zeros([Xstar_r.shape[0], SP.prod(self.t)])
                c_diag = list()
                for j in range(self.out_dims):
                    temp = SP.dot(self.basis[j] ,SP.dot(KV['K_c'][j], self.basis[j].T))
                    c_diag.append(SP.diag(temp))
                C = reduce(SP.kron, c_diag)                
                R_star_star = self.covar_r[0].K(hyperparams['covar_r'][0], Xstar_r)
                R_tr_star = self.covar_r[0].Kcross(hyperparams['covar_r'][0])
                r = SP.dot(R_tr_star.T, USUr)
                q = unravel(SP.kron(C,SP.diag(R_star_star)),Ystar_covar.shape[0],Ystar_covar.shape[1])
                t = SP.zeros(Ystar_covar.shape)
                printProgressBar(0, Xstar_r.shape[0], prefix = 'Computing perdiction varaince:', suffix = 'Complete', length = 20) 
                if self.out_dims == 1:
                    for j in range(self.t[0]):
                        USUC = usuc[0][:,j]
                        temp = SP.kron(USUC, r) 
                        t[j,] = SP.sum((1./S * temp).T * temp.T, axis = 0)
                        printProgressBar(j+1, self.t[0] , prefix = 'Computing perdiction varaince:', suffix = 'Complete', length = 20)
                elif self.out_dims == 2:
                    for j in range(self.t[0]):
                        for k in range(self.t[1]):
                            USUC = reduce(SP.kron,[usuc[0][:,j],usuc[1][:,k]])
                            temp = SP.kron(USUC, r) 
                            t[k + (j*self.t[1]),] = SP.sum((1./S * temp).T * temp.T, axis = 0)
                        printProgressBar(j+1, self.t[0] , prefix = 'Computing perdiction varaince:', suffix = 'Complete', length = 20)
                elif self.out_dims == 3: 
                   for j in range(self.t[0]):
                        for k in range(self.t[1]):
                            for l in range(self.t[2]):
                                USUC = reduce(SP.kron,[usuc[0][:,j],usuc[1][:,k],usuc[2][:,l]])
                                temp = SP.kron(USUC, r) 
                                t[:,l + k*self.t[2] + j*self.t[1]*self.t[2],] = SP.sum((1./S * temp).T * temp.T, axis = 0)
                        if (j + 1) % 5 == 0:
                            printProgressBar(j+1, self.t[0] , prefix = 'Computing perdiction varaince:', suffix = 'Complete', length = 20)
                Ystar_covar = q - t
            else:
                Ystar_covar = SP.zeros([Xstar_r.shape[0], SP.prod(self.t)])
                c_diag = list()
                for j in range(self.out_dims):
                    temp = SP.dot(self.basis[j] ,SP.dot(KV['K_c'][j], self.basis[j].T))
                    c_diag.append(SP.diag(temp))
                C = reduce(SP.kron, c_diag)
                printProgressBar(0, Xstar_r.shape[0], prefix = 'Computing perdiction varaince:', suffix = 'Complete', length = 20) 
                for i in range(Xstar_r.shape[0]):
                    R_star_star = self.covar_r[0].K(hyperparams['covar_r'][0], SP.expand_dims(Xstar_r[i,:],axis=0))
                    self.covar_r[0].Xcross = SP.expand_dims(Xstar_r[i,:],axis=0)
                    R_tr_star = self.covar_r[0].Kcross(hyperparams['covar_r'][0])
                    r = SP.dot(R_tr_star.T, USUr)
                    q = C * R_star_star
                    t = SP.zeros([SP.prod(self.t)])
                    if self.out_dims == 1:
                        for j in range(self.t[0]):
                            USUC = usuc[0][:,j]
                            temp = SP.kron(USUC, r) 
                            t[j,] = SP.sum((1./S * temp).T * temp.T, axis = 0)
                    elif self.out_dims == 2:
                        for j in range(self.t[0]):
                            for k in range(self.t[1]):
                                USUC = reduce(SP.kron,[usuc[0][:,j],usuc[1][:,k]])
                                temp = SP.kron(USUC, r) 
                                t[k + (j*self.t[1]),] = SP.sum((1./S * temp).T * temp.T, axis = 0)
                    elif self.out_dims == 3: 
                       for j in range(self.t[0]):
                            for k in range(self.t[1]):
                                for l in range(self.t[2]):
                                    USUC = reduce(SP.kron,[usuc[0][:,j],usuc[1][:,k],usuc[2][:,l]])
                                    temp = SP.kron(USUC, r) 
                                    t[l + k*self.t[2] + j*self.t[1]*self.t[2],] = SP.sum((1./S * temp).T * temp.T, axis = 0)
                    Ystar_covar[i,:] = q - t
                    if (i + 1) % 10 == 0:
                        printProgressBar(i+1, Xstar_r.shape[0], prefix = 'Computing perdiction varaince:', suffix = 'Complete', length = 20)
                self.covar_r[0].Xcross = Xstar_r      
                
        return Ystar, Ystar_covar

    def _update_evd(self,hyperparams,covar_id):
        keys = []
        if 'covar_%s'%covar_id in hyperparams:
            keys.append('covar_%s'%covar_id)
        if 'X_%s'%covar_id in hyperparams:
            keys.append('X_%s'%covar_id)
            
        if not(self._is_cached(hyperparams,keys=keys)):
            K = []
            S = []
            U = []
            i=0
            for covar in getattr(self,'covar_%s'%covar_id):
                temp_kernel = covar.K(hyperparams['covar_%s'%covar_id][i])
                temp_kernel =  temp_kernel + SP.eye(temp_kernel.shape[0],temp_kernel.shape[1]) * 0.001 # Rosolving possible numerical instability
                temp_kernel[SP.isnan(temp_kernel)] = SP.finfo(SP.float32).eps
                temp_kernel[~SP.isfinite(temp_kernel)] = 1E6
                K.append(temp_kernel)
                S_temp,U_temp = LA.eigh(temp_kernel)
                S_temp[S_temp<=0] = SP.finfo(SP.float32).eps # Rosolving possible numerical instability
                S.append(S_temp)
                U.append(U_temp)
                i += 1
                
            self._covar_cache['K_%s'%covar_id] = K
            self._covar_cache['U_%s'%covar_id] = U
            self._covar_cache['S_%s'%covar_id] = S
            
    def get_covariances(self,hyperparams):
        """
        INPUT:
        hyperparams:  dictionary
        OUTPUT: dictionary with the fields
        Kr:     kernel on rows
        Kc:     kernel on columns
        Knoise: noise kernel
        """
        
        if self._is_cached(hyperparams):
            return self._covar_cache
        if self._covar_cache==None:
            self._covar_cache = {}

        # get EVD of C
        self._update_evd(hyperparams,'c')
        self._update_evd(hyperparams,'r')
        self._update_evd(hyperparams,'s')
        self._update_evd(hyperparams,'o')
        
        # create short-cut
        KV = self._covar_cache

        keys = list(set(hyperparams.keys()) & set(['covar_c','covar_s']))
        if not(self._is_cached(hyperparams,keys=keys)):
            Usi_c = list()
            Usi_s = list()
            Ktilde_c = list()
            Ktilde_s = list()
            Stilde_c = list()
            Utilde_c = list()
            Stilde_s = list()
            Utilde_s = list()
            for i in range(self.out_dims):
                Usi_c.append((SP.sqrt(1./KV['S_c'][i]) * KV['U_c'][i]).T)
                Usi_s.append((SP.sqrt(1./KV['S_s'][i]) * KV['U_s'][i]).T)
                Ktilde_c.append(SP.dot(Usi_s[i], SP.dot(self.nbasis[i].T, SP.dot(self.basis[i], SP.dot(KV['K_c'][i], 
                                       SP.dot(self.basis[i].T, SP.dot(self.nbasis[i] ,Usi_s[i].T)))))))
                S_temp,U_temp = LA.eigh(Ktilde_c[i])
                Stilde_c.append(S_temp)
                Utilde_c.append(U_temp)                
                Ktilde_s.append(SP.dot(Usi_c[i], SP.dot(self.basis[i].T, SP.dot(self.nbasis[i], SP.dot(KV['K_s'][i], 
                                       SP.dot(self.nbasis[i].T, SP.dot(self.basis[i] ,Usi_c[i].T)))))))
                S_temp,U_temp = LA.eigh(Ktilde_s[i])
                Stilde_s.append(S_temp)
                Utilde_s.append(U_temp)
            
            KV['USi_c'] = Usi_c
            KV['USi_s'] = Usi_s
            KV['Ktilde_c'] = Ktilde_c; KV['Utilde_c'] = Utilde_c; KV['Stilde_c'] = Stilde_c
            KV['Ktilde_s'] = Ktilde_s; KV['Utilde_s'] = Utilde_s; KV['Stilde_s'] = Stilde_s

        keys = list(set(hyperparams.keys()) & set(['covar_r','covar_o']))
        if not(self._is_cached(hyperparams,keys=keys)):
            Usi_r = list()
            Usi_o = list()
            Ktilde_r = list()
            Ktilde_o = list()
            Stilde_r = list()
            Utilde_r = list()
            Stilde_o = list()
            Utilde_o = list()
            for i in range(self.in_dims):
                Usi_r.append((SP.sqrt(1./KV['S_r'][i]) * KV['U_r'][i]).T)
                Usi_o.append((SP.sqrt(1./KV['S_o'][i]) * KV['U_o'][i]).T)
                Ktilde_r.append(SP.dot(Usi_o[i],SP.dot(KV['K_r'][i],Usi_o[i].T)))
                S_temp,U_temp = LA.eigh(Ktilde_r[i])
                Stilde_r.append(S_temp)
                Utilde_r.append(U_temp)                
                
                Ktilde_o.append(SP.dot(Usi_r[i],SP.dot(KV['K_o'][i],Usi_r[i].T)))
                S_temp,U_temp = LA.eigh(Ktilde_o[i])
                Stilde_o.append(S_temp)
                Utilde_o.append(U_temp)
                
            KV['USi_r'] = Usi_r
            KV['USi_o'] =  Usi_o            
            KV['Ktilde_r'] = Ktilde_r; KV['Utilde_r'] = Utilde_r; KV['Stilde_r'] = Stilde_r
            KV['Ktilde_o'] = Ktilde_o; KV['Utilde_o'] = Utilde_o; KV['Stilde_o'] = Stilde_o
     
        KV['UYtildeU_rc'] = reduce(SP.dot,[reduce(SP.kron,KV['Utilde_r']).T,reduce(SP.kron,KV['USi_o']),SP.reshape(self.Y,[self.n,SP.prod(self.t)]),
                            reduce(SP.kron,self.nbasis),reduce(SP.kron,KV['USi_s']).T,reduce(SP.kron,KV['Utilde_c'])])  
        KV['UYtildeU_os'] = reduce(SP.dot,[reduce(SP.kron,KV['Utilde_o']).T,reduce(SP.kron,KV['USi_r']),SP.reshape(self.Y,[self.n,SP.prod(self.t)]),
                            reduce(SP.kron,self.basis),reduce(SP.kron,KV['USi_c']).T,reduce(SP.kron,KV['Utilde_s'])])  
        
        KV['Stilde_rc'] = SP.kron(reduce(SP.kron,KV['Stilde_c']),reduce(SP.kron,KV['Stilde_r'])) + 1
        KV['Stilde_os'] = SP.kron(reduce(SP.kron,KV['Stilde_s']),reduce(SP.kron,KV['Stilde_o'])) + 1
        
        KV['hyperparams'] = copy.deepcopy(hyperparams)
        return KV
        

    def _LML_covar(self,hyperparams):
        """
        log marginal likelihood
        """

        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            pdb.set_trace()
            LG.error('linalg exception in _LML_covar')
            return 1E6
        
        Si = 1./KV['Stilde_rc']
        lml_quad = 0.5 * (ravel(KV['UYtildeU_rc'])**2 * Si).sum()
        lml_det = 0.5 * (SP.log(reduce(SP.kron,KV['S_s'])).sum() * self.n + SP.log(reduce(SP.kron,KV['S_o'])).sum() * SP.prod(self.t))
        lml_det += 0.5 * SP.log(KV['Stilde_rc']).sum()
        lml_const = 0.5 * self.nt * (SP.log(2 * SP.pi))
        
        lml = lml_quad + lml_det + lml_const

        return lml

    def LMLgrad(self,hyperparams):
        """
        evaluates the gradient of the log marginal likelihood
        
        Input:
        hyperparams: dictionary
        """
        
        RV = {}
   
        # gradient with respect to covariance C
        if 'covar_c' in hyperparams or 'X_c' in hyperparams:
            RV.update(self._LMLgrad_c(hyperparams))
        
        # gradient with respect to covariance R
        if 'covar_r' in hyperparams or 'X_r' in hyperparams:
            RV.update(self._LMLgrad_r(hyperparams))

        # gradient with respect oc covariace Sigma
        if 'covar_s' in hyperparams or 'X_s' in hyperparams:
            RV.update(self._LMLgrad_s(hyperparams))

        # gradient with respect to covariance Omega
        if 'covar_o' in hyperparams or 'X_o' in hyperparams:
            RV.update(self._LMLgrad_o(hyperparams))
            
        if self.prior!=None:
            priorRV = self.prior.LMLgrad(hyperparams)
            for key in priorRV.keys():
                if key in RV:
                    RV[key] += priorRV[key]
                else:
                    RV[key] = priorRV[key]
      
        return RV


    def _LMLgrad_c(self,hyperparams):
        """
        evaluates the gradient with respect to the covariance matrix C
        """
        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LMLgrad_x_c')
            return {'X_c':SP.zeros(hyperparams['X_c'].shape)}

        Si = 1./KV['Stilde_rc']
        Yhat = unravel(Si * ravel(KV['UYtildeU_rc']),self.n, SP.prod(self.nbn))
        RV = {}
        
        if 'covar_c' in hyperparams:
            k=0
            RV['covar_c'] = []
            for covar in self.covar_c:
                theta = SP.zeros(len(hyperparams['covar_c'][k]))
                USU = SP.dot(self.basis[k].T, SP.dot(self.nbasis[k], SP.dot(KV['USi_s'][k].T,KV['Utilde_c'][k])))
                for i in range(len(theta)):
                    Kgrad_c = covar.Kgrad_theta(hyperparams['covar_c'][k],i)
                    UdKU = SP.dot(USU.T,SP.dot(Kgrad_c,USU))
                    temp = copy.deepcopy(KV['Stilde_c'])
                    for z in range(len(temp)):
                        temp[z] = SP.diag(temp[z])
                    temp[k] = UdKU
                    SUdKUS = reduce(SP.kron,temp)
                    LMLgrad_det = SP.sum(Si*SP.kron(SP.diag(SUdKUS),reduce(SP.kron,KV['Stilde_r'])))
                    SYUdKU = SP.dot(SUdKUS,reduce(SP.kron,KV['Stilde_r']) * Yhat.T)  # may be optimized
                    LMLgrad_quad = -(Yhat.T*SYUdKU).sum()
                    LMLgrad = 0.5*(LMLgrad_det + LMLgrad_quad)
                    theta[i] = LMLgrad
                RV['covar_c'].append(theta)
                k+=1
                
        return RV


    def _LMLgrad_r(self,hyperparams):
        """
        evaluate gradients with respect to covariance matrix R
        """
        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LMLgrad_x_r')
            return {'X_r':SP.zeros(hyperparams['X_r'].shape)}

        Si = 1./KV['Stilde_rc']
        Yhat = unravel(Si * ravel(KV['UYtildeU_rc']),self.n,SP.prod(self.nbn))
        RV = {}
                
        if 'covar_r' in hyperparams:
            k=0
            RV['covar_r'] = []
            for covar in self.covar_r:
                theta = SP.zeros(len(hyperparams['covar_r'][k]))
                USU = SP.dot(KV['USi_o'][k].T,KV['Utilde_r'][k])
                for i in range(len(theta)):
                    Kgrad_r = covar.Kgrad_theta(hyperparams['covar_r'][k],i)
                    UdKU = SP.dot(USU.T,SP.dot(Kgrad_r,USU))
                    temp = copy.deepcopy(KV['Stilde_r'])
                    for z in range(len(temp)):
                        temp[z] = SP.diag(temp[z])
                    temp[k] = UdKU
                    SUdKUS = reduce(SP.kron,temp)
                    LMLgrad_det = SP.sum(Si*SP.kron(reduce(SP.kron,KV['Stilde_c']),SP.diag(SUdKUS)))
                    SYUdKU = SP.dot(SUdKUS,Yhat*reduce(SP.kron,KV['Stilde_c']))
                    LMLgrad_quad = -(Yhat*SYUdKU).sum()
                    LMLgrad = 0.5*(LMLgrad_det + LMLgrad_quad)
                    theta[i] = LMLgrad
                RV['covar_r'].append(theta)
                k+=1
                
        return RV


    def _LMLgrad_s(self,hyperparams):
        """
        evaluate gradients with respect to covariance matrix Sigma
        """
        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LMLgrad_x_sigma')
            return {'X_s':SP.zeros(hyperparams['X_s'].shape)}

        Si = 1./KV['Stilde_os']
        Yhat = unravel(Si * ravel(KV['UYtildeU_os']),self.n,SP.prod(self.bn))
        RV = {}

        if 'covar_s' in hyperparams:
            k=0
            RV['covar_s'] = []
            for covar in self.covar_s:
                theta = SP.zeros(len(hyperparams['covar_s'][k]))                
                USU = SP.dot(self.nbasis[k].T, SP.dot(self.basis[k], SP.dot(KV['USi_c'][k].T,KV['Utilde_s'][k])))

                for i in range(len(theta)):
                    Kgrad_s = covar.Kgrad_theta(hyperparams['covar_s'][k],i)
                    UdKU = SP.dot(USU.T,SP.dot(Kgrad_s,USU))
                    temp = copy.deepcopy(KV['Stilde_s'])
                    for z in range(len(temp)):
                        temp[z] = SP.diag(temp[z])
                    temp[k] = UdKU
                    SUdKUS = reduce(SP.kron,temp)
                    LMLgrad_det = SP.sum(Si*SP.kron(SP.diag(SUdKUS),reduce(SP.kron,KV['Stilde_o'])))
                    SYUdKU = SP.dot(SUdKUS,reduce(SP.kron,KV['Stilde_o']) * Yhat.T)                    
                    LMLgrad_quad = -(Yhat.T*SYUdKU).sum()
                    LMLgrad = 0.5*(LMLgrad_det + LMLgrad_quad)
                    theta[i] = LMLgrad
                RV['covar_s'].append(theta)
                k += 1
                
        return RV


    def _LMLgrad_o(self,hyperparams):
        """
        evaluates the gradient with respect to the covariance matrix Omega
        """
        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LMLgrad_x_omega')
            return {'X_o':SP.zeros(hyperparams['X_o'].shape)}

        Si = 1./KV['Stilde_os']
        Yhat = unravel(Si * ravel(KV['UYtildeU_os']),self.n,SP.prod(self.bn))
        RV = {}

        if 'covar_o' in hyperparams:
            k=0
            RV['covar_o'] = []
            for covar in self.covar_o:
                theta = SP.zeros(len(hyperparams['covar_o'][k]))
                USU = SP.dot(KV['USi_r'][k].T,KV['Utilde_o'][k])
                for i in range(len(theta)):
                    Kgrad_o = covar.Kgrad_theta(hyperparams['covar_o'][k],i)
                    UdKU = SP.dot(USU.T,SP.dot(Kgrad_o,USU))
                    temp = copy.deepcopy(KV['Stilde_o'])
                    for z in range(len(temp)):
                        temp[z] = SP.diag(temp[z])
                    temp[k] = UdKU
                    SUdKUS = reduce(SP.kron,temp)
                    LMLgrad_det = SP.sum(Si*SP.kron(reduce(SP.kron,KV['Stilde_s']),SP.diag(SUdKUS)))
                    SYUdKU = SP.dot(SUdKUS,Yhat*reduce(SP.kron,KV['Stilde_s']))
                    LMLgrad_quad = -(Yhat*SYUdKU).sum()
                    LMLgrad = 0.5*(LMLgrad_det + LMLgrad_quad)
                    theta[i] = LMLgrad
                RV['covar_o'].append(theta)
                k+=1
        
        return RV
    
    def _is_cached(self,hyperparams,keys=None):
        """ check wheter model parameters are cached"""
        if self._covar_cache is None:
            return False
        if not ('hyperparams' in self._covar_cache):
            return False
        if keys==None:
            keys = hyperparams.keys()
        for key in keys:
            for i in range(len(hyperparams[key])):
                for j in range(len(hyperparams[key][i])):
                    if (self._covar_cache['hyperparams'][key][i][j]!=hyperparams[key][i][j]).any():
                        return False
        return True