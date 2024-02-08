from Trefft_tools import Inner_term_PP, Gamma_term

from numpy import exp, dot, conj, sin, cos, sqrt, pi
from numpy.linalg import norm
import numpy as np 
from numpy import trapz as Int
from collections import namedtuple


# Numerical Versions
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

def test_inner():
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
    assert relative_error < 1E-5

def test_Gamma():
    P = np.array([0,1])
    Q = np.array([3,1])

    T = (Q - P)/norm(Q-P)
    N = np.array([0,1])

    Edge = namedtuple('Edge',['P','Q','N','T'])
    E = Edge(P,Q,N,T)

    k = 8.
    d_n = np.array([1,1])/norm([1,1])
    d_m = np.array([1,-1])/norm([1,-1])

    TestFunction = namedtuple('TestFunction',['d','k'])
    phi_n = TestFunction(d=d_n,k=k)
    psi_m = TestFunction(d=d_m,k=k)

    d1 = 0.5
    I_exact = Gamma_term(phi_n, psi_m, E, k, d1)
    I_num = num_Gamma( k, P, Q, N, d_n, d_m, d1=d1,  Nt=int(1E4))
    relative_error = abs(I_exact - I_num)/abs(I_exact)
    assert relative_error < 1E-5
