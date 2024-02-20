import numpy as np
from scipy.special import jn, hankel1

def sound_soft_freespace(X,Y,k, R=1, c=(0,0), theta_inc=0, M = 15):
    c_x, c_y = c
    r = np.sqrt( (X-c_x)**2 + (Y-c_y)**2)
    theta = np.arctan2(Y - c_y, X - c_x)
    n = np.expand_dims(np.arange(-M,M+1),[1,2])
    u_sc=  -np.sum(1j**n * jn(n,k*R) / hankel1(n,k*R) * hankel1( n, k*r) * np.exp(1j*n*(theta - theta_inc)), axis=0)
    u = np.where(r>R, u_sc, np.full_like(u_sc,np.nan))
    return u




def exact_sound_soft(X,Y,k, H, rad=1, c=(0,0), theta_inc=0, M = 15, N = 3):
    c_x, c_y = c
    u = sound_soft_freespace(X,Y,k, R=rad, c=c, theta_inc=theta_inc, M = M)
    c_up_old = (H + (H-c_y)) #top
    c = (c_x,c_up_old)
    u += sound_soft_freespace(X,Y,k, R=rad, c=c, theta_inc=theta_inc, M = M)

    c_down_old = (-H - (c_y+H)) #bottom
    c = (c_x,c_down_old)
    u += sound_soft_freespace(X,Y,k, R=rad, c=c, theta_inc=theta_inc, M = M)

    for n in range(N):
        c_up = (H + (H-c_down_old)) #top
        c = (c_x,c_up)
        u += sound_soft_freespace(X,Y,k, R=rad, c=c, theta_inc=theta_inc, M = M)
        c_down = (-H - (c_up_old+H)) #bottom
        c = (c_x,c_down)
        u += sound_soft_freespace(X,Y,k, R=rad, c=c, theta_inc=theta_inc, M = M)
        c_up_old = c_up
        c_down_old = c_down

    r = np.sqrt( (X-c_x)**2 + (Y-c_y)**2)
    u_inc= np.exp(1j*k*(X*np.cos(theta_inc) + Y*np.sin(theta_inc)))
    u = np.where(r>rad, u+u_inc, np.zeros_like(u))

    return u 
