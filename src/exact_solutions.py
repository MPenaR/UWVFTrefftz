from scipy.spatial.distance import cdist
from scipy.special import j0, y0
import numpy as np

def H01(x):
    return j0(x) + 1j*y0(x)

def GreenFunction(k, H, XY, x0, y0, M = 100):
    n = np.arange(-M,M+1)
    xy_0 = np.stack([np.full(2*M+1,fill_value=x0), 2*n*H + (1 - 2* (n%2) )*y0],axis=1)
    G = 1j/4*np.sum(H01(k*cdist(XY, xy_0)),axis=-1)
    return G
