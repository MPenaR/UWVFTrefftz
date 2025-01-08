r"""
module for implementing blocks of fluxes, i.e. matrices containing all the fluxes of the same type for a 
given pair of cells.
"""

from numpy_types import real_array, complex_array
from numpy import dot, exp, sinc, subtract, newaxis, pi
from geometry import Edge

def SoundHard_block(k : complex, edge : Edge, d : real_array, d_d : real_array, d_1 : float) -> complex_array: #, M_trig : real_array) -> complex_array:
    
    r"""
    Computes the block for an edge in a sound hard boundary.

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
    """

    l = edge.l
    N = edge.N
    T = edge.T
    M = edge.M



    I = -1j*k*l*dot(d, N)[:,newaxis]*exp(1j*k*dot(d_d,M))*sinc(k*l/(2*pi)*dot(d_d,T))*(1 + d_1*dot(d, N))
#    I = -1j*k*l*dot(d, N)[:,newaxis]*exp(1j*k*dot(d_d,M))*sinc(k*l/(2*pi)*dot(d_d,T))*(1 + d_1*dot(d, N))/ exp(1j*k*subtract.outer(dot(d,M_trig), dot(d,M_trig)).transpose()) 

    return I
