# module for evaluation of simple fluxes
from numpy import dot, sinc, pi, exp, sqrt
from FEM import TestFunction, TrialFunction
from numpy_types import real_array

from typing import NamedTuple
class Edge(NamedTuple):
    M : real_array
    N : real_array
    l : float 
    T : real_array


def Gamma_term( phi : TrialFunction, psi : TestFunction, k : float, edge : Edge, d_1 : float):
    r"""
    Computes the flux on a sound-hard boundary, that is:

    .. math::
    
        \int_E \phi_n(\mathbf{x})\overline{\psi_m(\mathbf{x})}\,\mathrm{d}\ell(\mathbf{x})
    
    Parameters
    ----------
    phi : TrialFunction
        Trial function.
    psi : TestFunction
        Test function.
    k : float
        Wave number.
    edge : Edge
        Edge parameters.
    d_1 : float
        Stabilyzing parameter.

    Returns
    -------
    I : complex
        The integral.
    
    """

    d_m = psi.d
    d_n = phi.d
    
    M = edge.M
    l = edge.l
    N = edge.N
    T = edge.T

    I = -1j*k*l*(1 + d_1 * dot(d_n, N))*dot(d_m, N)*exp(1j*k*dot(d_n - d_m, M)) * sinc(k*l/(2*pi)*dot(d_n-d_m, T))
    return I

def Inner_term(phi, psi, edge, k, a, b):
    """computes the flux on a facet with respect to the degrees
    of freedom from the same cell."""

    d_m = psi.d
    d_n = phi.d

    k_n = k * sqrt(phi.n)
    k_m = k * sqrt(psi.n)

    
    M = edge.M
    N = edge.N
    T = edge.T
    l = edge.l

    I = -1j*l/2*(2*a*k + k_n*dot(d_n, N) + k_m*dot(d_m, N) + 2*b/k*k_n*dot(d_n, N)*k_m*dot(d_m, N))*exp(1j*dot(k_n*d_n - k_m*d_m, M))*sinc(l/(2*pi)*dot(k_n*d_n - k_m*d_m,T))

    return I
