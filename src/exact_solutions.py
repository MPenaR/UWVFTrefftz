from scipy.spatial.distance import cdist
from scipy.special import j0, y0
import numpy as np

def H01(x):
    return j0(x) + 1j*y0(x)

def GreenFunctionImages(k, H, XY, x_0, y_0, M = 100):
    n = np.arange(-M,M+1)
    xy_0 = np.stack([np.full(2*M+1,fill_value=x_0), 2*n*H + (1 - 2* (n%2) )*y_0],axis=1)
    G = 1j/4*np.sum(H01(k*cdist(XY, xy_0)),axis=-1)
    return G

def GreenFunctionModes(k, H, XY, x_0, y_0, M = 20):
    n = np.arange(0,M+1)
    beta_n = np.emath.sqrt(k**2 - (n*np.pi/H)**2)
    norm = np.full_like(n,1/H)
    norm[0] = 1/2*H
    G = -np.sum( norm*np.exp(1j*np.outer(np.abs(XY[:,0] - x_0),beta_n))/(2*1j*beta_n)*np.cos(np.pi*np.outer(XY[:,1],n)/H)*np.cos(n*np.pi*y_0/H), -1)
    return G
