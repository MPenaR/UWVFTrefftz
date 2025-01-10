r"""
Module for the evaluation of fluxes for a single pair of test and trial functions. It was the initial
implementation of the code and now serves as a test for the vectorized implementation. It is tested
against numerical computations of the fluxes.

All around this page, :math:`\Phi_{n}(\mathbf{x})=\exp(ik\mathbf{x}\cdot\mathbf{d}_n)` and :math:`\Psi_{m}(\mathbf{x})=\exp(ik\mathbf{x}\cdot\mathbf{d}_m)` 
ar the trial and test functions. :math:`l` is the length of an edge, :math:`\mathbf{M}` its midpoint and :math:`\boldsymbol{\tau}` and :math:`\mathbf{n}` 
its tangent and normal unitary vectors. 
"""



from numpy import dot, sinc, pi, exp, sqrt, conj
from FEM import TestFunction, TrialFunction
from geometry import Edge


def SoundHard( phi : TrialFunction, psi : TestFunction, k : float, edge : Edge, d_1 : float) -> complex:
    r"""
    Computes the flux on a sound-hard boundary, that is:

    .. math::
    
        \int_{E}\left(\Phi_n(\mathbf{x})+\frac{d_{1}}{ik}\nabla \Phi_n(\mathbf{x})\cdot\mathbf{n}\right)\overline{\nabla \Psi_m(\mathbf{x})\cdot\mathbf{n}}\,\mathrm{d}S_{\mathbf{x}}

    This quantity can be exactly evaluated as:

    .. math::
    
        \boxed{-ikl\left(1+d_{1}\mathbf{d}_{n}\cdot\mathbf{n}\right)\mathbf{d}_{m}\cdot\mathbf{n}e^{ik\left(\mathbf{d}_{n}-\mathbf{d}_{m}\right)\cdot\mathbf{M}}\mathrm{sinc}\left(\frac{kl}{2\pi}\left(\mathbf{d}_{n}-\mathbf{d}_{m}\right)\cdot\boldsymbol{\tau}\right)}


    Parameters
    ----------
    phi : TrialFunction
        Trial function.
    psi : TestFunction
        Test function.
    k : float
        Wavenumber.
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

def Inner(phi : TrialFunction, psi : TestFunction, edge : Edge, k : float, a : float, b : float) -> complex:
    r"""
    Computes the flux on a inner facet with respect to the degrees
    of freedom from the same cell, that is:
    
    FORGOT TO INCLUDE IT.    

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
    a : float
        Stabilyzing parameter.
    b : float
        Stabilyzing parameter.

    Returns
    -------
    I : complex
        The integral.


    """

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


def Radiating_local(phi : TrialFunction, psi : TestFunction, k : float, edge : Edge, d_2 : float) -> complex:
    r"""
    Computes the flux on a radiating boundary with respect to the degrees
    of freedom from the same cell, that is:

    TODO: it is assuming that the radiating boundary consists of a vertical segment. This should be easy to generalize.
    
    .. math::
    
        -\int_{E}\left(d_{2}ik\Phi_n(\mathbf{x})+\nabla \Phi_n(\mathbf{x})\cdot\mathbf{n}\right)\overline{\Psi_n(\mathbf{x})}\,\mathrm{d}S_{\mathbf{x}}

    which can be computed as:

    .. math::
    
        \boxed{-ikl\left(d_{2}+\mathbf{d}_{n}\cdot\mathbf{n}\right)e^{ik\left(\mathbf{d}_{n}-\mathbf{d}_{m}\right)\cdot\mathbf{M}}\mathrm{sinc}\left(\frac{kl}{2\pi}\left(\mathbf{d}_{n}-\mathbf{d}_{m}\right)\mathbf{j}\right)}

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
    d_2 : float
        Stabilyzing parameter.

    Returns
    -------
    I : complex
        The integral.
    
    """

    d_n = phi.d
    d_m = psi.d

    l = edge.l
    M = edge.M
    N = edge.N
    T = edge.T


    I = -1j*k*l*(d_2 + dot(d_n, N))*exp(1j*k*dot(d_n - d_m, M))*sinc(k*l/(2*pi)*dot(d_n-d_m, T))

    return I


def Radiating_nonlocal(phi : TrialFunction, psi : TestFunction, k : float, edge_u : Edge, edge_v : Edge, d_2 : float, N_modes : int, H : float) -> complex:
    r"""
    Computes the flux on a radiating boundary with respect to the degrees
    of freedom from another cell, that is:

    TODO: it is assuming that the radiating boundary consists of a vertical segment. This should be easy to generalize.
    
    Parameters
    ----------
    phi : TrialFunction
        Trial function.
    psi : TestFunction
        Test function.
    k : float
        Wave number.
    edge_u : Edge
        Edge of the triangle associated to the trial function.
    edge_v : Edge
        Edge of the triangle associated to the test function.
    d_2 : float
        Stabilyzing parameter.
    N_modes : int
        Number of modes for the approximation of the NtD map.
    H : float
        height of the waveguide. 

    Returns
    -------
    I : complex
        The integral.

    
    """
    d_n = phi.d
    d_m = psi.d
     
    l_u = edge_u.l
    M_u = edge_u.M
    l_v = edge_v.l
    M_v = edge_v.M


    N = edge_u.N
    T = edge_u.T

    I1 = -1j*k*H*dot(d_n,N)*dot(d_m,N)*d_2*exp(1j*k*(dot(d_n,M_u) - dot(d_m,M_v)))*l_u/H*l_v/H*(
        sinc(k*l_u/(2*pi)*d_n[1])*sinc(k*l_v/(2*pi)*d_m[1]) + 1/2*sum( [ k**2 / abs(sqrt(complex(k**2 - (s*pi/H)**2)))**2 * (
        exp( 1j*s*pi/H*M_u[1])*sinc(k*l_u/(2*pi)*d_n[1] + s*l_u/(2*H)) + exp(-1j*s*pi/H*M_u[1])*sinc(k*l_u/(2*pi)*d_n[1] - s*l_u/(2*H)) ) *(
        exp(-1j*s*pi/H*M_v[1])*sinc(k*l_v/(2*pi)*d_m[1] + s*l_v/(2*H)) + exp( 1j*s*pi/H*M_v[1])*sinc(k*l_v/(2*pi)*d_m[1] - s*l_v/(2*H)) )
        for s in range(1,N_modes)]) )
    
    I2 = -1j*k*H*dot(d_n,N)*(dot(d_m,N)-d_2)*exp(1j*k*(dot(d_n,M_u) - dot(d_m,M_v)))*l_u/H*l_v/H*(
        sinc(k*l_u/(2*pi)*d_n[1])*sinc(k*l_v/(2*pi)*d_m[1]) + 1/2*sum( [ k / sqrt(complex(k**2 - (s*pi/H)**2)) * (
        exp( 1j*s*pi/H*M_u[1])*sinc(k*l_u/(2*pi)*d_n[1] + s*l_u/(2*H)) + exp(-1j*s*pi/H*M_u[1])*sinc(k*l_u/(2*pi)*d_n[1] - s*l_u/(2*H)) ) *(
        exp(-1j*s*pi/H*M_v[1])*sinc(k*l_v/(2*pi)*d_m[1] + s*l_v/(2*H)) + exp( 1j*s*pi/H*M_v[1])*sinc(k*l_v/(2*pi)*d_m[1] - s*l_v/(2*H)) )
        for s in range(1,N_modes)]) )
    
    I3 = 1j*k*H*dot(d_m,N)*d_2*exp(1j*k*(dot(d_n,M_u) - dot(d_m,M_v)))*l_u/H*l_v/H*(
        sinc(k*l_u/(2*pi)*d_n[1])*sinc(k*l_v/(2*pi)*d_m[1]) + 1/2*sum( [ k / conj(sqrt(complex(k**2 - (s*pi/H)**2))) * (
        exp( 1j*s*pi/H*M_u[1])*sinc(k*l_u/(2*pi)*d_n[1] + s*l_u/(2*H)) + exp(-1j*s*pi/H*M_u[1])*sinc(k*l_u/(2*pi)*d_n[1] - s*l_u/(2*H)) ) *(
        exp(-1j*s*pi/H*M_v[1])*sinc(k*l_v/(2*pi)*d_m[1] + s*l_v/(2*H)) + exp( 1j*s*pi/H*M_v[1])*sinc(k*l_v/(2*pi)*d_m[1] - s*l_v/(2*H)) )
        for s in range(1,N_modes)]) )

    return  I1 + I2 + I3

