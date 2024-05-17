from scipy.spatial.distance import cdist
from scipy.special import j0, y0
import numpy as np
from numpy import cos, pi, exp, abs
from numpy.lib.scimath import sqrt
def H01(x):
    return j0(x) + 1j*y0(x)

def GreenFunctionImages(k, H, XY, x_0, y_0, M = 100):
    n = np.arange(0,M)
    minus_1_to_n = 1 - 2* (n%2) 
    xy_0p = np.stack([np.full(M,fill_value=x_0), 2*H*np.ceil(n/2) + minus_1_to_n*y_0],axis=1)
    xy_0m = np.stack([np.full(M,fill_value=x_0), -2*H*np.ceil(n/2) - minus_1_to_n*y_0],axis=1)
    xy_0 = np.vstack([xy_0p, xy_0m])
    G = 1j/4*np.sum(H01(k*cdist(XY, xy_0)),axis=-1)
    return G

def GreenFunctionModes(k, H, XY, x_0, y_0, M = 20):
    n = np.arange(0,M)
    beta_n = sqrt(k**2 - (n*np.pi/H)**2)
#    norm = np.full_like(n,2/H)
    norm = np.full(M, 2./H, dtype=np.int64)
    norm[0] = 1./H 
    G = -np.sum( norm*exp(1j*np.outer( abs(XY[:,0] - x_0),beta_n)) / (2*1j*beta_n) * cos( pi*np.outer(XY[:,1],n)/H) * cos(n* pi*y_0/H), -1)
    return G
