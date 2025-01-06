# module for evaluation of simple fluxes
from numpy import dot, sinc, pi, exp

def Gamma_term(phi, psi, k, edge, d_1):
    """Computes the flux on a sound-hard boundary"""

    d_m = psi.d
    d_n = phi.d
    
    M = edge.M
    l = edge.l
    N = edge.N
    T = edge.T

    I = -1j*k*l*(1 + d_1 * dot(d_n, N))*dot(d_m, N)*exp(1j*k*dot(d_n - d_m, M)) * sinc(k*l/(2*pi)*dot(d_n-d_m, T))
    return I