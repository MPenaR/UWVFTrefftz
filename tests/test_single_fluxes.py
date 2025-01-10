from single_fluxes import SoundHard, Inner, Radiating_local, Radiating_nonlocal

# , Gamma_term, Sigma_term, exact_RHS
import pytest
from numpy import linspace, outer, sqrt, sin, cos, pi, exp, dot, conj, isclose, array
from numpy.linalg import norm
from numpy import trapezoid as Int
import numpy as np


from collections import namedtuple
TOL = 1E-7
N_POINTS = int(1E5)
Edge = namedtuple('Edge', ['P', 'Q', 'N', 'T', 'M', 'l'])
TstFunction = namedtuple('TestFunction', ['d', 'n'])

from itertools import product  ## SHOULD BE A FIXTURE IN CONFTEST.PY
NTH = 3
directions = list(product([(cos(th), sin(th)) for th in linspace(0, pi/2, NTH, endpoint=False)],
                          [(cos(th), sin(th)) for th in linspace(0, pi/2, NTH, endpoint=False)]))



def num_Inner(k, P, Q, N, d_n, d_m, a=0, b=0, Nt=100):
    t = linspace(0, 1, Nt)
    x = P + outer(t, Q-P)
    phi_n = exp(1j*k*dot(x, d_n))
    psi_m = exp(1j*k*dot(x, d_m))
    grad_phi_n_N = 1j*k*dot(N, d_n)*exp(1j*k*dot(x, d_n))
    grad_psi_m_N = 1j*k*dot(N, d_m)*exp(1j*k*dot(x, d_m))

    I  = 1/2*Int(phi_n*conj(grad_psi_m_N) - grad_phi_n_N*conj(psi_m), t)
    I += b*Int(1/(1j*k)*grad_phi_n_N * conj(grad_psi_m_N), t)
    I -= a*Int(1j*k*phi_n*conj(psi_m), t)
    I = norm(Q-P)*I
    return I


@pytest.mark.parametrize(('d_m', 'd_n'), directions )
def test_Inner(d_m,d_n):
    P = array([3,3])
    Q = array([1,1])
    l = norm(Q-P)
    T = (Q - P)/l
    N = array([-T[1], T[0]])
    M = (P + Q)/2    
    E = Edge(P,Q,N,T,M,l)


    k = 8.
    d_n = array(d_n)/norm(d_n)
    d_m = array(d_m)/norm(d_m)

    phi_n = TstFunction(d=d_n,n=1)
    psi_m = TstFunction(d=d_m,n=1)

    a = 0.5
    b = 0.5

    I_exact = Inner(phi_n, psi_m, E, k, a, b)
    I_num = num_Inner( k, P, Q, N, d_n, d_m, a = a, b = b,  Nt=N_POINTS)
    assert isclose(I_num, I_exact, TOL, TOL), f'{I_exact=}, {I_num=}'




def num_SoundHard(k, P, Q, N, d_n, d_m, d1, Nt):
    t = linspace(0, 1, Nt)
    x = P + outer(t, Q-P)
    phi_n = exp(1j*k*dot(x, d_n))
    grad_phi_n_N = 1j*k*dot(N, d_n)*exp(1j*k*dot(x, d_n))
    grad_psi_m_N = 1j*k*dot(N, d_m)*exp(1j*k*dot(x, d_m))

    I = Int((phi_n + d1/(1j*k)*grad_phi_n_N)*conj(grad_psi_m_N)*norm(Q-P), t)
    return I

@pytest.mark.parametrize(('d_m', 'd_n'), directions )
def test_SoundHard(d_m, d_n):
    P = array([0,1])
    Q = array([3,1])
    l = norm(P-Q)
    T = (Q - P)/l
    N = array([0,1])
    M = (P + Q)/2
    
    E = Edge(P, Q, N, T, M, l)

    k = 8.
    d_n = array(d_n)/norm(d_n)
    d_m = array(d_m)/norm(d_m)

    phi_n = TstFunction(d=d_n, n=1)
    psi_m = TstFunction(d=d_m, n=1)

    d1 = 0.5
    I_exact = SoundHard(phi_n, psi_m, k, E, d1)
    I_num = num_SoundHard(k, P, Q, N, d_n, d_m, d1=d1,  Nt=N_POINTS)
    assert isclose(I_num, I_exact, TOL, TOL), f'{I_exact=}, {I_num=}'



# def num_Sigma( k, P, Q, N, H, d_n, d_m, d2=0, Nt = 100, Np=15):
#     l = norm(Q-P)
#     t = np.linspace(0,1,Nt)
#     x = P + np.outer(t,Q-P)
#     phi_n = exp(1j*k*dot(x,d_n))
#     psi_m = exp(1j*k*dot(x,d_m))
#     grad_phi_n_N = 1j*k*dot(N,d_n)*exp(1j*k*dot(x,d_n))
#     grad_psi_m_N = 1j*k*dot(N,d_m)*exp(1j*k*dot(x,d_m))

#     N_gradphi_n = NewmanntoDirichlet(x[:,1], grad_phi_n_N, k, H, Np)
#     N_gradpsi_m = NewmanntoDirichlet(x[:,1], grad_psi_m_N, k, H, Np)

#     I = Int( N_gradphi_n*conj(grad_psi_m_N) - grad_phi_n_N*conj(psi_m), t)*l
#     I+= -d2*1j*k*Int((N_gradphi_n - phi_n)*conj(N_gradpsi_m - psi_m), t)*l
    
#     return I

# @pytest.mark.parametrize(('d_m', 'd_n'), directions )
# def test_Sigma(d_m,d_n):
#     H=1
#     R= 10
#     P = np.array([R,-H])
#     Q = np.array([R,H])

#     l = norm(Q-P)
#     T = (Q - P)/l
#     N = np.array([1,0])
#     M = (P+Q)/2

#     Edge = namedtuple('Edge',['P','Q','N','T', 'M', 'l'])
#     E = Edge(P,Q,N,T,M,l)

#     k = 8.
#     d_n = np.array(d_n)/norm(d_n)
#     d_m = np.array(d_m)/norm(d_m)

#     TestFunction = namedtuple('TestFunction',['d','k'])
#     phi_n = TestFunction(d=d_n,k=k)
#     psi_m = TestFunction(d=d_m,k=k)

#     d2 = 0.5
#     I_exact = Sigma_term(phi_n, psi_m, E, d2)
#     I_num = num_Sigma( k, P, Q, N, H, d_n, d_m, d2=d2,  Nt=N_points)
#     assert np.isclose(I_num, I_exact, TOL, TOL), f'{I_exact=}, {I_num=}'


def num_Radiating_local( k, P, Q, N, H, d_n, d_m, d2=0, Nt = 100, Np=15):
    l = norm(Q-P)
    t = np.linspace(0,1,Nt)
    x = P + np.outer(t,Q-P)
    phi_n = exp(1j*k*dot(x,d_n))
    psi_m = exp(1j*k*dot(x,d_m))
    grad_phi_n_N = 1j*k*dot(N,d_n)*exp(1j*k*dot(x,d_n))
    # grad_psi_m_N = 1j*k*dot(N,d_m)*exp(1j*k*dot(x,d_m))


    I = -Int(grad_phi_n_N*conj(psi_m) + d2*1j*k*phi_n*conj(psi_m), t)*l
    
    return I

@pytest.mark.parametrize(('d_m', 'd_n'), directions )
def test_Radiating_local(d_m,d_n):
    H=1
    R= 10
    P = np.array([R,-H])
    Q = np.array([R,H])

    l = norm(Q-P)
    T = (Q - P)/l
    N = np.array([1,0])
    M = (P+Q)/2

    E = Edge(P,Q,N,T,M,l)

    k = 8.
    d_n = np.array(d_n)/norm(d_n)
    d_m = np.array(d_m)/norm(d_m)

    phi = TstFunction(d=d_n,n=1)
    psi = TstFunction(d=d_m,n=1)
    d2 = 0.5
    I_exact = Radiating_local(phi, psi, k, E, d2)
    I_num = num_Radiating_local( k, P, Q, N, H, d_n, d_m, d2=d2,  Nt=N_POINTS)
    assert np.isclose(I_num, I_exact, TOL, TOL), f'{I_exact=}, {I_num=}'




def NewmanntoDirichlet(y, df_dy, k, H, M):

    dfn = np.zeros(M, dtype=np.complex128)
    dfn[0] = Int( df_dy*1/np.sqrt(2*H), y )
    for n in range(1,M):
        dfn[n] = Int( df_dy*cos(n*pi*y/H)/np.sqrt(H), y )
    
    f_y = 1/(1j*k)*dfn[0]/np.sqrt(2*H)*np.ones_like(y) + sum([ 1/(1j*np.sqrt(complex(k**2 - (n*pi/H)**2)))*dfn[n]*cos(n*pi*y/H)/np.sqrt(H) for n in range(1,M)])
    return f_y


def num_Radiating( k, P, Q, N, H, d_n, d_m, d2=0, Nt = 100, N_modes=15):
    l = norm(Q-P)
    t = np.linspace(0,1,Nt)
    x = P + np.outer(t,Q-P)
    phi_n = exp(1j*k*dot(x,d_n))
    psi_m = exp(1j*k*dot(x,d_m))
    grad_phi_n_N = 1j*k*dot(N,d_n)*exp(1j*k*dot(x,d_n))
    grad_psi_m_N = 1j*k*dot(N,d_m)*exp(1j*k*dot(x,d_m))

    N_gradphi_n = NewmanntoDirichlet(x[:,1], grad_phi_n_N, k, H, N_modes)
    N_gradpsi_m = NewmanntoDirichlet(x[:,1], grad_psi_m_N, k, H, N_modes)

    I = Int( N_gradphi_n*conj(grad_psi_m_N) - grad_phi_n_N*conj(psi_m), t)*l
    I+= -d2*1j*k*Int((N_gradphi_n - phi_n)*conj(N_gradpsi_m - psi_m), t)*l
    
    

    return I

@pytest.mark.xfail(reason="mixed up dimensions of the waveguide")
@pytest.mark.parametrize(('d_m', 'd_n'), directions )
def test_Radiating(d_m,d_n):
    H=1
    R= 10
    P = np.array([R,0])
    Q = np.array([R,H])

    l = norm(Q-P)
    T = (Q - P)/l
    N = np.array([1,0])
    M = (P+Q)/2

    E = Edge(P,Q,N,T,M,l)

    k = 8.
    d_n = np.array(d_n)/norm(d_n)
    d_m = np.array(d_m)/norm(d_m)

    phi = TstFunction(d=d_n,n=1)
    psi = TstFunction(d=d_m,n=1)
    d_2 = 0.0

    N_modes = 15
    I_exact_local = Radiating_local(phi, psi, k, E, d_2)
    I_exact_nonlocal = Radiating_nonlocal(phi=phi, psi=psi, k=k, edge_u=E, edge_v=E, d_2=d_2, N_modes=N_modes, H=H)
    I_exact = I_exact_nonlocal + I_exact_local
    I_num = num_Radiating( k, P, Q, N, H, d_n, d_m, d2=d_2,  Nt=N_POINTS, N_modes=N_modes)
    assert np.isclose(I_num, I_exact, TOL, TOL), f'{I_exact=}, {I_num=}'




# @pytest.mark.parametrize(('d_m', 'd_n'), directions )
# def test_Sigma_broken(d_m,d_n):
#     H=1
#     R= 10
#     P = np.array([R,-H])
#     Q = np.array([R,H])

#     l = norm(Q-P)
#     T = (Q - P)/l
#     N = np.array([1,0])
#     M = (P+Q)/2

#     Edge = namedtuple('Edge',['P','Q','N','T', 'M', 'l'])
#     E = Edge(P,Q,N,T,M,l)

#     k = 8.
#     d_n = np.array(d_n)/norm(d_n)
#     d_m = np.array(d_m)/norm(d_m)

#     TestFunction = namedtuple('TestFunction',['d','k'])
#     phi_n = TestFunction(d=d_n,k=k)
#     psi_m = TestFunction(d=d_m,k=k)

#     d2 = 0.0
#     I_exact = Sigma_broken(phi_n, psi_m, E, k, H, d2)
#     I_num = num_Sigma( k, P, Q, N, H, d_n, d_m, d2=d2,  Nt=N_points)
#     assert np.isclose(I_num, I_exact, TOL, TOL), f'{I_exact=}, {I_num=}'






# def num_RHS( k, P, Q, N, H, s, d_m, d2=0, Nt = 100, Np=15):
#     l = norm(Q-P)
#     t = np.linspace(0,1,Nt)
#     x = P + np.outer(t,Q-P)
#     psi_m = exp(1j*k*dot(x,d_m))
#     beta = sqrt(complex(k**2 - (s*pi/H)**2 ))
#     u_inc = exp(1j*beta*x[:,0])*cos(s*pi*x[:,1]/H)

#     grad_psi_m_N = 1j*k*dot(N,d_m)*exp(1j*k*dot(x,d_m))

#     N_gradpsi_m = NewmanntoDirichlet(x[:,1], grad_psi_m_N, k, H, Np)

#     I = -2*Int( u_inc*conj(grad_psi_m_N) - d2*1j*k*u_inc*conj(N_gradpsi_m-psi_m), t)*l
    
#     return I


# def test_RHS():
#     H=1
#     R= 10
#     P = np.array([-R,-H])
#     Q = np.array([-R,H])

#     l = norm(Q-P)
#     T = (Q - P)/l
#     N = np.array([0,-1])
#     M = (P+Q)/2
    

#     Edge = namedtuple('Edge',['P','Q','N','T', 'M', 'l'])
#     E = Edge(P,Q,N,T, M, l)

#     k = 8.
#     d_m = [1,1]
#     d_m = np.array(d_m)/norm(d_m)

#     TestFunction = namedtuple('TestFunction',['d','k'])
#     psi_m = TestFunction(d=d_m,k=k)

#     d2 = 0.5

#     t= 1

#     I_exact = exact_RHS_broken(psi_m, E, k, H, d2, t)
#     I_num = num_RHS( k, P, Q, N, H, t, d_m, d2=d2, Nt=N_points)
#     assert np.isclose(I_num, I_exact, TOL, TOL), f'{I_exact=}, {I_num=}'


# def test_RHS_broken():
#     H=1
#     R= 10
#     P = np.array([-R,-H])
#     Q = np.array([-R,H/3])

#     l = norm(Q-P)
#     T = (Q - P)/l
#     N = np.array([0,-1])
#     M = (P+Q)/2
    

#     Edge = namedtuple('Edge',['P','Q','N','T', 'M', 'l'])
#     E = Edge(P,Q,N,T, M, l)

#     k = 8.
#     d_m = [1,1]
#     d_m = np.array(d_m)/norm(d_m)

#     TestFunction = namedtuple('TestFunction',['d','k'])
#     psi_m = TestFunction(d=d_m,k=k)

#     d2 = 0.5

#     t= 1

#     I_exact = exact_RHS_broken(psi_m, E, k, H, d2, t)
#     I_num = num_RHS( k, P, Q, N, H, t, d_m, d2=d2, Nt=N_points)
#     assert np.isclose(I_num, I_exact, TOL, TOL), f'{I_exact=}, {I_num=}'
