import numpy as np

# using truncated multivariate normal
# https://stackoverflow.com/questions/20115917/truncated-multivariate-normal-in-scipy
import emcee

from numpy.linalg import inv, cholesky
from scipy.linalg import qr
from scipy.stats import truncnorm

def lnprob_trunc_norm(x, mean, bounds, C):
    if np.any(x < bounds[:,0]) or np.any(x > bounds[:,1]):
        return -np.inf
    else:
        return -0.5*(x-mean).dot(inv(C)).dot(x-mean)

def multivariate_trunc_normal(bounds, mean, cov=None, size=1):
    """
    Usage:
    
    ```
    ndim = 10
    mean = (np.random.rand(ndim)-1)*3
    bounds=np.ones((ndim, 2))
    bounds[:, 0] = -3
    bounds[:, 1] = 3
    C = np.eye(ndim)
    theta = multivariate_trunc_normal(mean, C, bounds)
    ```
    
    If covariance is not specificed, defaults to identity matrix.
    """
    if cov is not None and len(cov.shape) == 1:
        ndim = mean.shape[0]
        nwalkers = max(10*ndim, size)
        nsteps = 1000 + size

        S = emcee.EnsembleSampler(nwalkers, ndim, lnprob_trunc_norm, args = (mean, bounds, cov))
        pos = emcee.utils.sample_ball(mean, np.sqrt(np.diag(cov)), size=nwalkers)
        pos, prob, state = S.run_mcmc(pos, nsteps)

        return pos[-size:].flatten()
    elif cov is None:
        ndim = mean.shape[0]
        cov = np.eye(ndim)
        return multivariate_trunc_normal(bounds, mean, cov, size)
    else:
        ndim = mean.shape[0]
        cov = np.diag(cov)
        return multivariate_trunc_normal(bounds, mean, cov, size)
        
        
        
        
        
        
        
        
        
        
        
        
        
