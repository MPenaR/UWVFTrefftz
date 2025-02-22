# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: UWVFTrefftz--hy3v2Qt
#     language: python
#     name: python3
# ---

from numpy import exp, dot, conj, sin, cos, sqrt, pi
from numpy.linalg import norm
import numpy as np 
import matplotlib.pyplot as plt
from numpy import trapz as Int
from collections import namedtuple


# # Inner terms

# First we are going to check the inner terms. There are $4$ situations:
#  - $\phi^+_n$ and $\psi^+_m$: 
# $$
# \frac{1}{2}\int_E \phi_n\overline{\nabla \psi_m\cdot\mathbf{n}}-\nabla \phi_n\cdot\mathbf{n}\overline{\psi_m}\,\mathrm{d}S_x+a\dots+b\dots
# $$
#  - $\phi^+_n$ and $\psi^-_m$: 
# $$
# -\frac{1}{2}\int_E \phi_n\overline{\nabla \psi_m\cdot\mathbf{n}}-\nabla \phi_n\cdot\mathbf{n}\overline{\psi_m}\,\mathrm{d}S_x+a\dots+b\dots
# $$
#  - $\phi^-_n$ and $\psi^+_m$: 
# $$
# \frac{1}{2}\int_E \phi_n\overline{\nabla \psi_m\cdot\mathbf{n}}-\nabla \phi_n\cdot\mathbf{n}\overline{\psi_m}\,\mathrm{d}S_x+a\dots+b\dots
# $$
#  - $\phi^-_n$ and $\psi^-_m$: 
# $$
# -\frac{1}{2}\int_E \phi_n\overline{\nabla \psi_m\cdot\mathbf{n}}-\nabla \phi_n\cdot\mathbf{n}\overline{\psi_m}\,\mathrm{d}S_x++a\dots+b\dots
# $$
#

# For checking the central fluxes only one expression needs to be checked: 
#
# $$
# \frac{1}{2}\int_E \varphi_n\overline{\nabla \psi_m\cdot\mathbf{n}}-\nabla \varphi_n\cdot\mathbf{n}\overline{\psi_m}\,\mathrm{d}S_x
# $$
#
# Assuming $\varphi_n(\mathbf{x})=e^{ik\mathbf{d}_n\cdot\mathbf{x}}$ and $\psi_m(\mathbf{x})=e^{ik\mathbf{d}_m\cdot\mathbf{x}}$, the term above becomes:
# $$
# -ikl\left(\mathbf{d}_{m}+\mathbf{d}_{n}\right)\cdot\mathbf{n}\frac{e^{ik\left(\mathbf{d}_{n}-\mathbf{d}_{m}\right)\cdot\mathbf{P}}}{2}\qquad\text{if}\quad\mathbf{d}_n\cdot\boldsymbol{\tau}=\mathbf{d}_m\cdot\boldsymbol{\tau}
# $$
# $$
# -\frac{\left(\mathbf{d}_{m}+\mathbf{d}_{n}\right)\cdot\mathbf{n}}{\left(\mathbf{d}_{n}-\mathbf{d}_{m}\right)\cdot\boldsymbol{\tau}}\frac{e^{ik\left(\mathbf{d}_{n}-\mathbf{d}_{m}\right)\cdot\mathbf{Q}}-e^{ik\left(\mathbf{d}_{n}-\mathbf{d}_{m}\right)\cdot\mathbf{P}}}{2}\qquad\text{otherwise}
# $$

# ## Exact term

def Inner_term_PP(phi, psi, edge, k, a, b):

    d_m = psi.d
    d_n = phi.d
    
    P = edge.P
    Q = edge.Q
    N = edge.N
    T = edge.T

    l = norm(Q-P)

    tol = 1E-6

    I = dot( d_m, N) + dot( d_n, N) + 2*b*dot( d_m, N)*dot( d_n, N) + 2*a


    if np.isclose( dot(d_m,T), dot(d_n,T), tol) :
        return -1/2*1j*k*l * I * exp(1j*k*dot(d_n - d_m, P))
    else:
        return -1/2*I/dot(d_n - d_m, T)*( exp(1j*k*dot(d_n - d_m, Q)) - exp(1j*k*dot(d_n - d_m, P)))


# ## Numerical term

# +
def num_inner( k, P, Q, N, d_n, d_m, a=0, b=0, Nt = 100):
    Px, Py = P[0], P[1]
    Qx, Qy = Q[0], Q[1]
    l = norm(Q-P)
    t = np.linspace(0,1,Nt)
    x = P + np.outer(t,Q-P)
    phi_n = exp(1j*k*dot(x,d_n))
    psi_m = exp(1j*k*dot(x,d_m))
    grad_phi_n_N = 1j*k*dot(N,d_n)*exp(1j*k*dot(x,d_n))
    grad_psi_m_N = 1j*k*dot(N,d_m)*exp(1j*k*dot(x,d_m))

    I = l/2*Int(phi_n*conj(grad_psi_m_N) - grad_phi_n_N*conj(psi_m), t)
    I+= l*b*Int(1/(1j*k)*grad_phi_n_N * conj(grad_psi_m_N), t)
    I-= l*a*Int(1j*k*phi_n*conj(psi_m),t)


    return I


# -

# ### They check

# +
# Test
P = np.array([3,3])
Q = np.array([1,1])

T = (Q - P)/norm(Q-P)
N = np.array([-T[1], T[0]])

Edge = namedtuple('Edge',['P','Q','N','T'])
E = Edge(P,Q,N,T)


k = 8.
d_n = np.array([1,1])/norm([1,1])
d_m = np.array([1,-1])/norm([1,-1])

TestFunction = namedtuple('TestFunction',['d','k'])
phi_n = TestFunction(d=d_n,k=k)
psi_m = TestFunction(d=d_m,k=k)

a = 0.5
b = 0.5


I_exact = Inner_term_PP(phi_n, psi_m, E, k, a, b)
I_num = num_inner( k, P, Q, N, d_n, d_m, a = a, b = b,  Nt=int(1E4))
relative_error = abs(I_exact - I_num)/abs(I_exact)
print(f'I_exact: {I_exact:.16f}\nI_num:   {I_num:.16f}\nRelative error: {relative_error :.2e}')


# -

# # Gamma terms

def Gamma_term(phi, psi, edge, k, d_1):

    d_m = psi.d
    d_n = phi.d
    
    P = edge.P
    Q = edge.Q
    N = edge.N
    T = edge.T

    l = norm(Q-P)

    I = (1 + d_1 * dot(d_n, N))*dot(d_m, N)

    if np.isclose( dot(d_m,T), dot(d_n,T), 1E-3) :
        return -1j*k*l* I * exp(1j*k*dot(d_n - d_m, P))
    else:
        return -I / dot(d_n - d_m, T) * ( exp(1j*k*dot(d_n - d_m, Q)) - exp(1j*k*dot(d_n - d_m, P)))


def num_Gamma( k, P, Q, N, d_n, d_m, d1=0, Nt = 100):
    Px, Py = P[0], P[1]
    Qx, Qy = Q[0], Q[1]
    l = norm(Q-P)
    t = np.linspace(0,1,Nt)
    x = P + np.outer(t,Q-P)
    phi_n = exp(1j*k*dot(x,d_n))
    psi_m = exp(1j*k*dot(x,d_m))
    grad_phi_n_N = 1j*k*dot(N,d_n)*exp(1j*k*dot(x,d_n))
    grad_psi_m_N = 1j*k*dot(N,d_m)*exp(1j*k*dot(x,d_m))

    I = Int( (phi_n + d1/(1j*k)*grad_phi_n_N)*conj(grad_psi_m_N)*l, t)
    return I


# ### They check

# +
P = np.array([0,1])
Q = np.array([3,1])

T = (Q - P)/norm(Q-P)
N = np.array([0,1])

Edge = namedtuple('Edge',['P','Q','N','T'])
E = Edge(P,Q,N,T)


d1 = 0.5
I_exact = Gamma_term(phi_n, psi_m, E, k, d1)
I_num = num_Gamma( k, P, Q, N, d_n, d_m, d1=d1,  Nt=int(1E4))
relative_error = abs(I_exact - I_num)/abs(I_exact)
print(f'I_exact: {I_exact:.16f}\nI_num:   {I_num:.16f}\nRelative error: {relative_error :.2e}')


# -

# # Sigma Terms

def Sigma_term(phi, psi, edge, k, H, d_2, Np = 15):

    d_n = phi.d
    d_m = psi.d

    d_mx = d_m[0]
    d_my = d_m[1]
    d_nx = d_n[0]
    d_ny = d_n[1]
    


    kH = k*H
    
    P = edge.P 
    N = edge.N
    x  = P[0]/H

    d_nN = dot(d_n,N)
    d_mN = dot(d_m,N)
    
    #first-like terms
    I1 = -2*1j*kH*exp(1j*(d_nx-d_mx)*kH*x)*((1-d_2)*d_mN*d_nN + d_2*(d_mN + d_nN))

    if np.isclose(d_ny, 0, 1E-3) and np.isclose(d_my, 0, 1E-3):
        F = I1
    elif np.isclose(d_ny, 0, 1E-3):
        F = I1 * sin(d_my*kH) / (d_my*kH)
    elif np.isclose(d_my, 0, 1E-3):
        F =  I1 * sin(d_ny*kH) / (d_ny*kH)
    else:
        I2 = -1j*kH*exp(1j*(d_nx-d_mx)*kH*x)*(1-d_2)*d_mN*d_nN * \
              sum([kH/sqrt(complex(kH**2 - (s*pi)**2)) * (sin(d_ny*kH+s*pi)/(d_ny*kH+s*pi) + sin(d_ny*kH-s*pi)/(d_ny*kH-s*pi)) 
                                                       * (sin(d_my*kH+s*pi)/(d_my*kH+s*pi) + sin(d_my*kH-s*pi)/(d_my*kH-s*pi))  
                                                         for s in range(1,Np)])
        
        F  = I1 * sin(d_my*kH) / (d_my*kH) * sin(d_ny*kH) / (d_ny*kH) + I2

    #second-like terms
        
    I = -2*1j*kH*(d_nN-d_2)*exp(1j*(d_nx-d_mx)*kH*x)
    if np.isclose(d_ny, d_my, 1E-3):
        S = I *exp(1j*(d_ny-d_my)*kH)
    else:
        S = I * sin((d_ny-d_my)*kH) / ((d_ny-d_my)*kH)  

    return F+S, F , S


# +
def NewmanntoDirichlet(y, df_dy, k, H, M):

    dfn = np.zeros(M, dtype=np.complex128)
    dfn[0] = Int( df_dy*1/sqrt(2*H), y )
    for n in range(1,M):
        dfn[n] = Int( df_dy*cos(n*pi*y/H)/sqrt(H), y )
    
    f_y = 1/(1j*k)*dfn[0]/sqrt(2*H)*np.ones_like(y) + sum([ 1/(1j*sqrt(complex(k**2 - (n*pi/H)**2)))*dfn[n]*cos(n*pi*y/H)/sqrt(H) for n in range(1,M)])
    return f_y


def num_Sigma( k, P, Q, N, H, d_n, d_m, d2=0, Nt = 100, Np=15):
    Px, Py = P[0], P[1]
    Qx, Qy = Q[0], Q[1]
    l = norm(Q-P)
    t = np.linspace(0,1,Nt)
    x = P + np.outer(t,Q-P)
    phi_n = exp(1j*k*dot(x,d_n))
    psi_m = exp(1j*k*dot(x,d_m))
    grad_phi_n_N = 1j*k*dot(N,d_n)*exp(1j*k*dot(x,d_n))
    grad_psi_m_N = 1j*k*dot(N,d_m)*exp(1j*k*dot(x,d_m))

    N_gradphi_n = NewmanntoDirichlet(x[:,1], grad_phi_n_N, k, H, Np)
    N_gradpsi_m = NewmanntoDirichlet(x[:,1], grad_psi_m_N, k, H, Np)

    I = Int( N_gradphi_n*conj(grad_psi_m_N) - grad_phi_n_N*conj(psi_m), t)*l
    I+= -d2*1j*k*Int((N_gradphi_n - phi_n)*conj(N_gradpsi_m - psi_m), t)*l
    return I


# +
H=1
R= 10
P = np.array([R,-H])
Q = np.array([R,H])

T = (Q - P)/norm(Q-P)
N = np.array([0,1])

Edge = namedtuple('Edge',['P','Q','N','T'])
E = Edge(P,Q,N,T)


d2 = 0.5
I_exact, _, _ = Sigma_term(phi_n, psi_m, E, k, H, d2)
I_num = num_Sigma( k, P, Q, N, H, d_n, d_m, d2=d2,  Nt=int(1E4))
relative_error = abs(I_exact - I_num)/abs(I_exact)
print(f'I_exact: {I_exact:.16f}\nI_num:   {I_num:.16f}\nRelative error: {relative_error :.2e}')
# -


