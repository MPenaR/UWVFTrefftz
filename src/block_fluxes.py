r"""
module for implementing blocks of fluxes, i.e. matrices containing all the fluxes of the same type for a 
given pair of cells.
"""

from numpy_types import real_array, complex_array
from numpy import dot, exp, sinc, subtract, add, newaxis, pi, outer
from geometry import Edge

def SoundHard_block(k : complex, edge : Edge, d : real_array, d_d : real_array, d_1 : float) -> complex_array: #, M_trig : real_array) -> complex_array:
    
    r"""
    Computes the block for an edge in a sound hard boundary. 
    That is it computes the matrix :math:`\mathbf{M}=(M_{mn})` with:    
    
    .. math::
    
        M_{mn}=\boxed{(-ikl\left(1+d_{1}\mathbf{d}_{n}\cdot\mathbf{n}\right)\mathbf{d}_{m}\cdot\mathbf{n}e^{ik\left(\mathbf{d}_{n}-\mathbf{d}_{m}\right)\cdot\mathbf{M}}\mathrm{sinc}\left(\frac{kl}{2\pi}\left(\mathbf{d}_{n}-\mathbf{d}_{m}\right)\cdot\boldsymbol{\tau}\right)}


    Parameters:
    -----------

    - k : complex
        Wavenumber
    - edge : Edge
        Edge
    - d : real_array
        Set of directions
    - d_d : real_array
        Nd x Nd x 2 "Matrix" of differences of directions.
    - d_1 : float
        Stabilyzing parameter.    
    """

    l = edge.l
    N = edge.N
    T = edge.T
    M = edge.M



    I = -1j*k*l*outer(dot(d, N), (1 + d_1*dot(d, N)))*exp(1j*k*dot(d_d, M))*sinc(k*l/(2*pi)*dot(d_d,T))

# this is with basis function normalization
#    I = -1j*k*l*dot(d, N)[:,newaxis]*exp(1j*k*dot(d_d,M))*sinc(k*l/(2*pi)*dot(d_d,T))*(1 + d_1*dot(d, N))/ exp(1j*k*subtract.outer(dot(d,M_trig), dot(d,M_trig)).transpose()) 

    return I

def Inner_block(k : complex, edge : Edge, d : real_array, a : float, b : float) -> complex_array:
    r"""
    Computes the block for an inner edge.

    Parameters:
    -----------

    - k : complex
        Wavenumber
    - edge : Edge
        Edge
    - d : real_array
        Set of directions
    - a : float
        Stabilyzing parameter.    
    - b : float
        Stabilyzing parameter.    
"""
    l = edge.l
    N = edge.N
    M = edge.M
    T = edge.T
    
    I = -1j*k*l*(a + add.outer(sqrt(n_m)*dot(d,N),sqrt(n_n)*dot(d,N))/2 + b*np.outer(sqrt(n_m)*dot(d,N),sqrt(n_n)*dot(d,N))) \
    *exp(-1j*k* subtract.outer(sqrt(n_m)*dot(d,M),sqrt(n_n)*dot(d,M)))                                             \
    *sinc(l*k/(2*pi)*subtract.outer(sqrt(n_m)*dot(d,T),sqrt(n_n)*dot(d,T)))/ np.exp(1j*k*np.subtract.outer(sqrt(n_n)*dot(d,M_n), conj(sqrt(n_m))*dot(d,M_m)).transpose())

    return I

