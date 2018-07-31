import scipy as SP
import sys
sys.path.append('../')

def init(method,Y,X_r,opts):
    """
    initialized hyperparams, bounds and filters
    """
    # init 
    if method=='STGPR':
        covar0_r = SP.random.rand(3)
        covar0_r/=covar0_r.sum()
        covar0_r=SP.squeeze(covar0_r)
        covar0_r = 0.5*SP.log(covar0_r)
        hyperparams = {'covar':covar0_r,'lik':0.5 * SP.log(SP.random.rand(1))}
        Ifilter = {'covar':SP.ones(covar0_r.size),'lik':SP.ones(hyperparams['lik'].size)}
        bounds = {'covar':SP.array([[-5,+5]]*covar0_r.size), 'lik':SP.array([[-5,+5]])}  
    elif method=='MTGPR':
        covar0_r = SP.random.rand(4)
        covar0_r /= covar0_r.sum()
        covar0_r = SP.squeeze(covar0_r)
        covar0_r = 0.5 * SP.log(covar0_r)   
        covar0_c = SP.random.rand(4)
        covar0_c/=covar0_c.sum()
        covar0_c=SP.squeeze(covar0_c)
        covar0_c = 0.5 * SP.log(covar0_c)
        lik0 = SP.random.rand(1)
        lik0 = 0.5 * SP.log(lik0)
        hyperparams = {'covar_c':covar0_c,'covar_r':covar0_r,'lik':lik0}
        Ifilter = {'covar_c':SP.ones(covar0_c.size),'covar_r':SP.ones(covar0_r.size),'lik':SP.ones(lik0.size)}
        bounds = {'covar_c':SP.array([[-10,+5]]*covar0_c.size), 'covar_r':SP.array([[-10,+5]]*covar0_r.size), 'lik':SP.array([[-10,+5]])}
    elif method=='Zero':
        covar0_r = SP.zeros(4)
        covar0_o = SP.zeros(1)
        covar0_c = list()
        covar0_s = list()
        for i in range(Y.ndim-1):
            temp_c = SP.zeros(4)
            covar0_c.append(temp_c)

            temp_s = SP.zeros(4)
            covar0_s.append(temp_s)
        hyperparams = {'covar_c':covar0_c,'covar_r':[covar0_r],'covar_s':covar0_s,'covar_o':[covar0_o]}
        Ifilter = None
        bounds = None
    return hyperparams,Ifilter,bounds