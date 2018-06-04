import logging as LG
import scipy as SP
import scipy.linalg as LA
import copy
from gpbase import GPBASE
import core.likelihood.likelihood_base as likelihood_base
from core.util.utilities import ravel, unravel, printProgressBar
    
class sMTGPR(GPBASE):

    __slots__ = ['covar_c','covar_r']
    
    def __init__(self, covar_r = None, covar_c = None, likelihood = None, prior = None, basis = None):
        assert isinstance(likelihood,likelihood_base.GaussIsoLik), 'likelihood is not implemented yet'
        self.covar_r = covar_r
        self.covar_c = covar_c
        self.likelihood = likelihood
        self.prior = prior
        self.basis = basis
        self._covar_cache = None
        self.Y = None
        

    def setData(self, Y, Y_hat, X = None, **kwargs):
        """
        set data
        Y:    Outputs [n x t]
        """
        assert Y.ndim == 2, 'Y must be a two dimensional vector'
        self.Y = Y
        self.Y_hat = Y_hat
        self.n = Y.shape[0]
        self.t = Y_hat.shape[1]
        self.nt = self.n * self.t
        if X is not None:
            self.covar_r.X = X
        self._invalidate_cache()
        
    def predict(self, hyperparams, Xstar_r, compute_cov = False):
        """
        predict on Xstar
        """
        KV = self.get_covariances(hyperparams)
        
        self.covar_r.Xcross = Xstar_r
        
        Kstar_r = self.covar_r.Kcross(hyperparams['covar_r'])
        Kstar_c = self.covar_c.K(hyperparams['covar_c'])  
        KinvY = SP.dot(KV['U_r'], SP.dot(KV['Ytilde'], KV['U_c'].T))
        BD = SP.dot(self.basis,Kstar_c)
        Ystar = reduce(SP.dot,[Kstar_r.T,KinvY, BD.T])
        
        Ystar_covar = []
        if compute_cov:           
            BDU = SP.dot(BD, KV['U_c'])
            BDB = (BD * self.basis).sum(-1)              
            Ystar_covar = SP.zeros([Xstar_r.shape[0], self.Y.shape[1]])
            s_rev = 1./KV['S']
            printProgressBar(0, BDU.shape[0], prefix = 'Computing perdiction varaince:', suffix = 'Complete', length = 20)
            if Xstar_r.shape[0] < 500:
                R_star_star = self.covar_r.K(hyperparams['covar_r'], Xstar_r)
                self.covar_r.Xcross = Xstar_r
                R_tr_star = self.covar_r.Kcross(hyperparams['covar_r'])
                RU = SP.dot(R_tr_star.T, KV['U_r'])
                q = unravel(SP.kron(BDB, SP.diag(R_star_star)),Ystar_covar.shape[0],Ystar_covar.shape[1])
                t = SP.zeros(Ystar_covar.shape)   
                for j in range(BDU.shape[0]):
                    temp = SP.kron(BDU[j,:], RU)
                    t[:,j] = SP.sum((s_rev * temp).T * temp.T, axis = 0)
                    if (j + 1) % 10000 == 0:
                        printProgressBar(j+1, BDU.shape[0] , prefix = 'Computing perdiction varaince:', suffix = 'Complete', length = 20)
                
                Ystar_covar = q - t
            else:    
                for i in range(Xstar_r.shape[0]):
                    R_star_star = self.covar_r.K(hyperparams['covar_r'], SP.expand_dims(Xstar_r[i,:],axis=0))
                    self.covar_r.Xcross = SP.expand_dims(Xstar_r[i,:],axis=0)
                    R_tr_star = self.covar_r.Kcross(hyperparams['covar_r'])
                    RU = SP.dot(R_tr_star.T, KV['U_r'])
                    q = SP.kron(BDB, R_star_star)
                    t = SP.zeros([self.basis.shape[0]])                
                    for j in range(BDU.shape[0]):
                        temp = SP.kron(BDU[j,:], RU) 
                        t[j,] = SP.sum((s_rev * temp).T * temp.T, axis = 0)
                    Ystar_covar[i,:] = q - t
                    if (i + 1) % (Xstar_r.shape[0]/10) == 0:
                        printProgressBar(i+1, Xstar_r.shape[0], prefix = 'Computing perdiction varaince:', suffix = 'Complete', length = 20)
                self.covar_r.Xcross = Xstar_r       
        return Ystar, Ystar_covar
  
    
    def get_covariances(self,hyperparams):
        
        if self._is_cached(hyperparams):
            return self._covar_cache
        if self._covar_cache==None:
            self._covar_cache = {}
            
        if not(self._is_cached(hyperparams,keys=['covar_c'])):
            K_c = self.covar_c.K(hyperparams['covar_c'])
            S_c,U_c = LA.eigh(K_c)
            self._covar_cache['K_c'] = K_c
            self._covar_cache['U_c'] = U_c
            self._covar_cache['S_c'] = S_c
        else:
            K_c = self._covar_cache['K_c']
            U_c = self._covar_cache['U_c']
            S_c = self._covar_cache['S_c']
            
        if not(self._is_cached(hyperparams,keys=['covar_r'])):
            K_r = self.covar_r.K(hyperparams['covar_r'])
            S_r,U_r = LA.eigh(K_r)
            self._covar_cache['K_r'] = K_r
            self._covar_cache['U_r'] = U_r
            self._covar_cache['S_r'] = S_r
        else:
            K_r = self._covar_cache['K_r']
            U_r = self._covar_cache['U_r']
            S_r = self._covar_cache['S_r']
        
        Binv = SP.linalg.pinv(self.basis.T).T       
        S = SP.kron(S_c,S_r) + self.likelihood.Kdiag(hyperparams['lik'],self.nt)
        UYUB = reduce(SP.dot,[U_r.T, self.Y, Binv.T, U_c])
        YtildeVec = (1./S) * ravel(UYUB)
        self._covar_cache['Binv'] = Binv
        self._covar_cache['S'] = S
        self._covar_cache['UYUB'] = UYUB
        self._covar_cache['Ytilde'] = unravel(YtildeVec,self.n,self.t)
        UBinv = SP.dot(U_c.T, Binv)
        self._covar_cache['UBinvB'] = SP.dot(UBinv, self.basis) 
        self._covar_cache['UBinvBinvU'] = SP.dot(UBinv, UBinv.T) 
        self._covar_cache['S_c_tilde'] = SP.diag(reduce(SP.dot, [self._covar_cache['UBinvB'], K_c, self._covar_cache['UBinvB'].T]))
        self._covar_cache['hyperparams'] = copy.deepcopy(hyperparams)
        return self._covar_cache
        

    def _LML_covar(self,hyperparams):
        """
        log marginal likelihood
        """
        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LML_covar')
            return 1E6
        except ValueError:
            LG.error('value error in _LML_covar')
            return 1E6
 
        lml_quad = 0.5*(KV['Ytilde']*KV['UYUB']).sum()
        lml_det =  0.5 * SP.log(KV['S']).sum()
        lml_const = 0.5*self.n*self.t*(SP.log(2*SP.pi))
        lml = lml_quad + lml_det + lml_const

        return lml

    def _LMLgrad_covar(self,hyperparams):
        """
        evaluates the gradient of the log marginal likelihood with respect to the
        hyperparameters of the covariance function
        """
        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LMLgrad_covar')
            return {'covar_r':SP.zeros(len(hyperparams['covar_r'])),'covar_c':SP.zeros(len(hyperparams['covar_c'])),'covar_r':SP.zeros(len(hyperparams['covar_r']))}
        except ValueError:
            LG.error('value error in _LMLgrad_covar')
            return {'covar_r':SP.zeros(len(hyperparams['covar_r'])),'covar_c':SP.zeros(len(hyperparams['covar_c'])),'covar_r':SP.zeros(len(hyperparams['covar_r']))}
 
        RV = {}
        Si = unravel(1./KV['S'],self.n,self.t)
    
        if 'covar_r' in hyperparams:
            theta = SP.zeros(len(hyperparams['covar_r']))
            for i in range(len(theta)):
                Kgrad_r = self.covar_r.Kgrad_theta(hyperparams['covar_r'],i)
                d = (KV['U_r']*SP.dot(Kgrad_r,KV['U_r'])).sum(0)                                
                LMLgrad_det = reduce(SP.dot, [d, Si, KV['S_c_tilde']])                
                UdKU = reduce(SP.dot, [KV['U_r'].T, Kgrad_r, KV['U_r']])
                SYUdKU = SP.dot(UdKU,(KV['Ytilde']*SP.tile(KV['S_c_tilde'][SP.newaxis,:],(self.n,1))))                
                LMLgrad_quad = - (KV['Ytilde']*SYUdKU).sum()
                LMLgrad = 0.5*(LMLgrad_det + LMLgrad_quad)
                theta[i] = LMLgrad          
            RV['covar_r'] = theta

        if 'covar_c' in hyperparams:
            theta = SP.zeros(len(hyperparams['covar_c']))
            for i in range(len(theta)):
                Kgrad_c = self.covar_c.Kgrad_theta(hyperparams['covar_c'],i)
                S_c_tilde_grad = reduce(SP.dot, [KV['UBinvB'], Kgrad_c, KV['UBinvB'].T])                
                LMLgrad_det = reduce(SP.dot, [KV['S_r'], Si, SP.diag(S_c_tilde_grad)])                
                SYUdKU = SP.dot((KV['Ytilde']*SP.tile(KV['S_r'][:,SP.newaxis],(1,self.t))),S_c_tilde_grad.T)
                LMLgrad_quad = -SP.sum(KV['Ytilde']*SYUdKU)
                LMLgrad = 0.5*(LMLgrad_det + LMLgrad_quad)
                theta[i] = LMLgrad
            RV['covar_c'] = theta

        return RV

    def _LMLgrad_lik(self,hyperparams):
        """
        evaluates the gradient of the log marginal likelihood with respect to
        the hyperparameters of the likelihood function
        """
        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LML_covar')
            return {'lik':SP.zeros(len(hyperparams['lik']))}
        except ValueError:
            LG.error('value error in _LML_covar')
            return {'lik':SP.zeros(len(hyperparams['lik']))}
        
        YtildeVec = ravel(KV['Ytilde'])             
        Kd_diag = self.likelihood.Kdiag_grad_theta(hyperparams['lik'],self.nt,0)
        LMLgrad_det = ((1./KV['S'])*Kd_diag).sum()
        LMLgrad_quad = -(YtildeVec**2 * Kd_diag).sum()
        LMLgrad = 0.5*(LMLgrad_det + LMLgrad_quad)
        return {'lik':SP.array([LMLgrad])}
