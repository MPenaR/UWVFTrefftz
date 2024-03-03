from Trefft_tools import Inner_term, Gamma_term, Sigma_term, exact_RHS


from numpy import exp, dot, conj, sin, cos, sqrt, pi
from numpy.linalg import norm
import numpy as np 
from numpy import trapz as Int
from collections import namedtuple
from itertools import product
import pytest


TOL = 1E-7
N_points = int(1E5)

NTH = 3
# directions
d_m = [(cos(th), sin(th)) for th in np.linspace(0,np.pi/2,NTH,endpoint=False)]
d_n = [(cos(th), sin(th)) for th in np.linspace(0,np.pi/2,NTH,endpoint=False)]

inputs = product(d_m,d_n)
# Numerical Versions
def num_inner( k, P, Q, N, d_n, d_m, a=0, b=0, Nt = 100):
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


@pytest.mark.parametrize(('d_m', 'd_n'), inputs )
def test_inner(d_m,d_n):
    P = np.array([3,3])
    Q = np.array([1,1])
    l = norm(Q-P)
    T = (Q - P)/l
    N = np.array([-T[1], T[0]])
    M = ( P + Q )/2    
    Edge = namedtuple('Edge',['P','Q','N','T','M','l'])
    E = Edge(P,Q,N,T,M,l)


    k = 8.
    d_n = np.array(d_n)/norm(d_n)
    d_m = np.array(d_m)/norm(d_m)

    TestFunction = namedtuple('TestFunction',['d','k'])
    phi_n = TestFunction(d=d_n,k=k)
    psi_m = TestFunction(d=d_m,k=k)

    a = 0.5
    b = 0.5

    I_exact = Inner_term(phi_n, psi_m, E, a, b)
    I_num = num_inner( k, P, Q, N, d_n, d_m, a = a, b = b,  Nt=N_points)
    relative_error = abs(I_exact - I_num)/abs(I_exact)
    assert relative_error < TOL, f'{I_exact=}, {I_num=}'




def num_Gamma( k, P, Q, N, d_n, d_m, d1=0, Nt = 100):
    l = norm(Q-P)
    t = np.linspace(0,1,Nt)
    x = P + np.outer(t,Q-P)
    phi_n = exp(1j*k*dot(x,d_n))
    grad_phi_n_N = 1j*k*dot(N,d_n)*exp(1j*k*dot(x,d_n))
    grad_psi_m_N = 1j*k*dot(N,d_m)*exp(1j*k*dot(x,d_m))

    I = Int( (phi_n + d1/(1j*k)*grad_phi_n_N)*conj(grad_psi_m_N)*l, t)
    return I

def test_Gamma():
    P = np.array([0,1])
    Q = np.array([3,1])
    l = norm(P-Q)
    T = (Q - P)/l
    N = np.array([0,1])
    M = ( P + Q)/2
    
    Edge = namedtuple('Edge',['P','Q','N','T','M','l'])
    E = Edge(P,Q,N,T,M,l)

    k = 8.
    d_n = np.array([1,1])/norm([1,1])
    d_m = np.array([1,-1])/norm([1,-1])

    TestFunction = namedtuple('TestFunction',['d','k'])
    phi_n = TestFunction(d=d_n,k=k)
    psi_m = TestFunction(d=d_m,k=k)

    d1 = 0.5
    I_exact = Gamma_term(phi_n, psi_m, E, d1)
    I_num = num_Gamma( k, P, Q, N, d_n, d_m, d1=d1,  Nt=N_points)
    relative_error = abs(I_exact - I_num)/abs(I_exact)
    assert relative_error < TOL





def NewmanntoDirichlet(y, df_dy, k, H, M):

    dfn = np.zeros(M, dtype=np.complex128)
    dfn[0] = Int( df_dy*1/sqrt(2*H), y )
    for n in range(1,M):
        dfn[n] = Int( df_dy*cos(n*pi*y/H)/sqrt(H), y )
    
    f_y = 1/(1j*k)*dfn[0]/sqrt(2*H)*np.ones_like(y) + sum([ 1/(1j*sqrt(complex(k**2 - (n*pi/H)**2)))*dfn[n]*cos(n*pi*y/H)/sqrt(H) for n in range(1,M)])
    return f_y


def num_Sigma( k, P, Q, N, H, d_n, d_m, d2=0, Nt = 100, Np=15):
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

def test_Sigma():
    H=1
    R= 10
    P = np.array([R,-H])
    Q = np.array([R,H])

    l = norm(Q-P)
    T = (Q - P)/l
    N = np.array([1,0])
    M = (P+Q)/2

    Edge = namedtuple('Edge',['P','Q','N','T', 'M', 'l'])
    E = Edge(P,Q,N,T,M,l)

    k = 8.
    d_n = [1,1]
    d_n = np.array(d_n)/norm(d_n)
    d_m = [1,-1]
    d_m = np.array(d_m)/norm(d_m)

    TestFunction = namedtuple('TestFunction',['d','k'])
    phi_n = TestFunction(d=d_n,k=k)
    psi_m = TestFunction(d=d_m,k=k)

    d2 = 0.5
    I_exact = Sigma_term(phi_n, psi_m, E, d2)
    I_num = num_Sigma( k, P, Q, N, H, d_n, d_m, d2=d2,  Nt=N_points)
    relative_error = abs(I_exact - I_num)/abs(I_exact)
    assert relative_error < TOL, f'{I_exact=}, {I_num=}'



def num_RHS( k, P, Q, N, H, s, d_m, d2=0, Nt = 100, Np=15):
    l = norm(Q-P)
    t = np.linspace(0,1,Nt)
    x = P + np.outer(t,Q-P)
    psi_m = exp(1j*k*dot(x,d_m))
    beta = sqrt(complex(k**2 - (s*pi/H)**2 ))
    u_inc = exp(1j*beta*x[:,0])*cos(s*pi*x[:,1]/H)

    grad_psi_m_N = 1j*k*dot(N,d_m)*exp(1j*k*dot(x,d_m))

    N_gradpsi_m = NewmanntoDirichlet(x[:,1], grad_psi_m_N, k, H, Np)

    I = -2*Int( u_inc*conj(grad_psi_m_N) - d2*1j*k*u_inc*conj(N_gradpsi_m-psi_m), t)*l
    
    return I

def test_RHS():
    H=1
    R= 10
    P = np.array([R,-H])
    Q = np.array([R,H])

    T = (Q - P)/norm(Q-P)
    N = np.array([0,1])

    Edge = namedtuple('Edge',['P','Q','N','T'])
    E = Edge(P,Q,N,T)

    k = 8.
    d_m = [1,1]
    d_m = np.array(d_m)/norm(d_m)

    TestFunction = namedtuple('TestFunction',['d','k'])
    psi_m = TestFunction(d=d_m,k=k)

    d2 = 0.5

    t= 1

    I_exact =exact_RHS(psi_m, E, k, H, d2, t)
    I_num = num_RHS( k, P, Q, N, H, t, d_m, d2=d2, Nt=N_points)
    relative_error = abs(I_exact - I_num)/abs(I_exact)
    assert relative_error < TOL

