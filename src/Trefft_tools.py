
import numpy as np
from numpy import dot, pi, exp, sqrt, sin, abs, conj
from numpy.linalg import norm
from collections import namedtuple
from labels import EdgeType
from scipy.sparse import coo_matrix, csr_matrix, spmatrix
from geometry_tools import Edge
from numpy import sinc, cos



TestFunction = namedtuple("TestFunction", ["k", "d"])

tfun_dtype = np.dtype()

class TrefftzSpace:
    '''Defines a finite dimensional Trefftz space given
    a mesh, the number of plane-waves per element and 
    the wave-numbers.

    It can create test and trial functions, aswell as 
    actual functions.
    '''

    def __init__( self, Omega, DOF_per_element : tuple[int], kappa : dict[str, float], th0=0 ):
        self.Omega = Omega
        self.N_trig = len(Omega.faces)
        self.kappa = np.zeros(self.N_trig, dtype=np.float64)
        self._elements = list(Omega.Elements())
        for e in Omega.Elements():
            self.kappa[e.faces[0].nr] = kappa[e.mat]
        
        
        if hasattr(DOF_per_element, '__iter__'):
            assert self.N_trig == len(DOF_per_element)
            self.local_N_DOF = np.array( DOF_per_element )
        else:
            self.local_N_DOF = np.full_like(self.kappa, fill_value=DOF_per_element, dtype=np.int32)
        self.N_DOF = np.sum(self.local_N_DOF)

        self.d = [ np.array([[ np.cos(th0 +th), 
                               np.sin(th0 +th)  ] for th in np.linspace(0, 2*pi, N, endpoint=False)]) for N in self.local_N_DOF ] 


        self.DOF_ownership = np.repeat( range(self.N_trig), self.local_N_DOF)
        self.DOF_start = np.cumsum(self.local_N_DOF) - self.local_N_DOF
        self.DOF_end = np.cumsum(self.local_N_DOF)
        self.DOF_range = [ list(range(s,e)) for (s,e) in zip(self.DOF_start,self.DOF_end)]
        self.global_to_local = np.array( [ n for N in self.local_N_DOF for n in range(N)])

    @property
    def TestFunctions( self ):
        return [ TestFunction( k= self.kappa[self.DOF_ownership[n]], d=self.d[self.DOF_ownership[n]][self.global_to_local[n]]) for n in range(self.N_DOF)]
    @property
    def TrialFunctions( self ):
        return [ TestFunction( k= self.kappa[self.DOF_ownership[n]], d=self.d[self.DOF_ownership[n]][self.global_to_local[n]]) for n in range(self.N_DOF)]




class TrefftzFunction:
    def __init__( self, V, DOFs = 0.):
        '''Returns a Trefftz function with degrees of freedom set to "DOFs"'''
        self.V = V
        self.DOFs = DOFs


    def Element(self, x, y ):
        e_ID = self.V.Omega(x,y).nr
        if e_ID == -1:
            return e_ID
        else:
            return self.V._elements[e_ID].faces[0].nr
    
    @property
    def DOFs( self ):
        return self._DOFs

    
    @DOFs.setter
    def DOFs( self, values):
        if hasattr(values, '__iter__'):
            assert self.V.N_DOF == len(values)
            self._DOFs = np.array(values)
        else:
            self._DOFs = np.full( self.V.N_DOF, values, dtype=np.complex128)


    def __call__(self, x, y ):
        e = self.Element(x,y)
        if e < 0: # (x,y) outside the mesh
            return np.nan
        
        k = self.V.kappa[e]
        P = self.DOFs[self.V.DOF_range[e]]
        D = self.V.d[e]

        r = np.array([x,y])
        y = sum( p*np.exp(1j*k*dot(d,r)) for (p,d) in zip(P,D) )
        return y 























def Gamma_term(phi, psi, edge, d_1):

    d_m = psi.d
    d_n = phi.d
    k = phi.k
    
    M = edge.M
    l = edge.l
    N = edge.N
    T = edge.T

    I = -1j*k*l*(1 + d_1 * dot(d_n, N))*dot(d_m, N)*exp(1j*k*dot(d_n - d_m,M))*sinc(k*l/(2*pi)*dot(d_n-d_m,T))
    return I


def Inner_term(phi, psi, edge, a, b):

    d_m = psi.d
    d_n = phi.d
    k = phi.k

    
    M = edge.M
    N = edge.N
    T = edge.T
    l = edge.l

    I = -1/2*1j*k*l*(dot(d_m,N) + dot(d_n,N) + 2*b*dot(d_m,N)*dot(d_n,N) + 2*a)*exp(1j*k*dot(d_n - d_m,M))*sinc(k*l/(2*pi)*dot(d_n-d_m,T))

    return I

def Inner_term_general(phi, psi, edge, n, a, b):

    d_m = psi.d
    d_n = phi.d
    k = phi.k

    
    M = edge.M
    N = edge.N
    T = edge.T
    l = edge.l

    I = -1/2*1j*k*l*(sqrt(n)*(dot(d_m,N) + dot(d_n,N)) + 2*b*n*dot(d_m,N)*dot(d_n,N) + 2*a)*exp(1j*k*sqrt(n)*dot(d_n - d_m,M))*sinc(k*sqrt(n)*l/(2*pi)*dot(d_n-d_m,T))

    return I

def sound_soft_term(phi, psi, edge, a):

    d_m = psi.d
    d_n = phi.d
    k = phi.k

    
    M = edge.M
    N = edge.N
    T = edge.T
    l = edge.l

    I = -1j*k*l*(dot(d_n, N) + a)* exp(1j*k*dot(d_n - d_m, M))*sinc(k*l/(2*pi)*dot(d_n-d_m,T))

    return  I


def Sigma_local(phi, psi, edge, k, H, d_2):

    d_n = phi.d
    d_m = psi.d
     
    l = edge.l
    M = edge.M
    N = edge.N
    T = edge.T


    I = -1j*k*l*(d_2 + dot(d_n,N))*exp(1j*k*dot(d_n - d_m,M))*sinc(k*l/(2*pi)*dot(d_n-d_m,T))

    return I


def Sigma_nonlocal(phi, psi, edge_u, edge_v, k, H, d_2, Np=15):

    d_n = phi.d
    d_m = psi.d
     
    l_u = edge_u.l
    M_u = edge_u.M
    l_v = edge_v.l
    M_v = edge_v.M


    N = edge_u.N
    T = edge_u.T

    I1 = -2j*k*H*dot(d_n,N)*dot(d_m,N)*d_2*exp(1j*k*(dot(d_n,M_u) - dot(d_m,M_v)))*l_u/(2*H)*l_v/(2*H)*(
        sinc(k*l_u/(2*pi)*d_n[1])*sinc(k*l_v/(2*pi)*d_m[1]) + 1/2*sum( [ k**2 / abs(sqrt(complex(k**2 - (s*pi/H)**2)))**2 * (
        exp( 1j*s*pi/H*M_u[1])*sinc(k*l_u/(2*pi)*d_n[1] + s*l_u/(2*H)) + exp(-1j*s*pi/H*M_u[1])*sinc(k*l_u/(2*pi)*d_n[1] - s*l_u/(2*H)) ) *(
        exp(-1j*s*pi/H*M_v[1])*sinc(k*l_v/(2*pi)*d_m[1] + s*l_v/(2*H)) + exp( 1j*s*pi/H*M_v[1])*sinc(k*l_v/(2*pi)*d_m[1] - s*l_v/(2*H)) )
        for s in range(1,Np)]) )
    
    I2 = 2j*k*H*dot(d_n,N)*(d_2-dot(d_m,N))*exp(1j*k*(dot(d_n,M_u) - dot(d_m,M_v)))*l_u/(2*H)*l_v/(2*H)*(
        sinc(k*l_u/(2*pi)*d_n[1])*sinc(k*l_v/(2*pi)*d_m[1]) + 1/2*sum( [ k / sqrt(complex(k**2 - (s*pi/H)**2)) * (
        exp( 1j*s*pi/H*M_u[1])*sinc(k*l_u/(2*pi)*d_n[1] + s*l_u/(2*H)) + exp(-1j*s*pi/H*M_u[1])*sinc(k*l_u/(2*pi)*d_n[1] - s*l_u/(2*H)) ) *(
        exp(-1j*s*pi/H*M_v[1])*sinc(k*l_v/(2*pi)*d_m[1] + s*l_v/(2*H)) + exp( 1j*s*pi/H*M_v[1])*sinc(k*l_v/(2*pi)*d_m[1] - s*l_v/(2*H)) )
        for s in range(1,Np)]) )
    
    I3 = 2j*k*H*dot(d_m,N)*d_2*exp(1j*k*(dot(d_n,M_u) - dot(d_m,M_v)))*l_u/(2*H)*l_v/(2*H)*(
        sinc(k*l_u/(2*pi)*d_n[1])*sinc(k*l_v/(2*pi)*d_m[1]) + 1/2*sum( [ k / conj(sqrt(complex(k**2 - (s*pi/H)**2))) * (
        exp( 1j*s*pi/H*M_u[1])*sinc(k*l_u/(2*pi)*d_n[1] + s*l_u/(2*H)) + exp(-1j*s*pi/H*M_u[1])*sinc(k*l_u/(2*pi)*d_n[1] - s*l_u/(2*H)) ) *(
        exp(-1j*s*pi/H*M_v[1])*sinc(k*l_v/(2*pi)*d_m[1] + s*l_v/(2*H)) + exp( 1j*s*pi/H*M_v[1])*sinc(k*l_v/(2*pi)*d_m[1] - s*l_v/(2*H)) )
        for s in range(1,Np)]) )

    return I1 + I2 + I3


def AssembleMatrix(V : TrefftzSpace, Edges : tuple[Edge], 
                   a = 0.5, b = 0.5, d_1 = 0.5, d_2 = 0.5, 
                   Np=10, full_matrix = False) -> spmatrix:
    '''Assembles de matrix for the bilinear form.
    a, b, d_1 and d_2 are the coefficients of the regularizing terms.
    Use full_matrix = Truee if the returned matrix should NOT be a sparse
    matrix.'''


    N_DOF = V.N_DOF

    values = []
    i_index = []
    j_index = []


    Side_edges = { EdgeType.SIGMA_L : [], EdgeType.SIGMA_R : []} 
    
    for E in Edges:
        match E.Type:
            case EdgeType.SIGMA_L | EdgeType.SIGMA_R:
                Side_edges[E.Type].append(E)
            case _:
                pass



    Phi = V.TrialFunctions
    Psi = V.TestFunctions # currently the same spaces 
    for E in Edges:
        match E.Type:
            case EdgeType.INNER:
                K_plus, K_minus = E.Triangles
                for n in V.DOF_range[K_plus]:
                    phi = Phi[n]
                    for m in V.DOF_range[K_plus]:
                        psi = Psi[m]
                        i_index.append(m)
                        j_index.append(n)
                        values.append(Inner_term(phi, psi, E, a, b))

                for n in V.DOF_range[K_minus]:
                    phi = Phi[n]
                    for m in V.DOF_range[K_plus]:
                        psi = Psi[m]
                        i_index.append(m)
                        j_index.append(n)
                        values.append(Inner_term(phi, psi, E, -a, -b))


                for n in V.DOF_range[K_plus]:
                    phi = Phi[n]
                    for m in V.DOF_range[K_minus]:
                        psi = Psi[m]
                        i_index.append(m)
                        j_index.append(n)
                        values.append(-Inner_term(phi, psi, E, a, b))

                for n in V.DOF_range[K_minus]:
                    phi = Phi[n]
                    for m in V.DOF_range[K_minus]:
                        psi = Psi[m]
                        i_index.append(m)
                        j_index.append(n)
                        values.append(-Inner_term(phi, psi, E, -a, -b))


            case EdgeType.GAMMA:
                K = E.Triangles[0]
                for m in V.DOF_range[K]:
                    psi = Psi[m]
                    for n in V.DOF_range[K]:
                        phi = Phi[n]
                        i_index.append(m)
                        j_index.append(n)
                        values.append(Gamma_term(phi, psi, E, d_1))
                    

            case EdgeType.D_OMEGA:
                K = E.Triangles[0]
                for m in V.DOF_range[K]:
                    psi = Psi[m]
                    for n in V.DOF_range[K]:
                        phi = Phi[n]
                        i_index.append(m)
                        j_index.append(n)
                        values.append(sound_soft_term(phi, psi, E, a))

            case EdgeType.SIGMA_L | EdgeType.SIGMA_R:
                K = E.Triangles[0]
                for n in V.DOF_range[K]:
                    phi = Phi[n]
                    for E_other in Side_edges[E.Type]:
                        K_other = E_other.Triangles[0]
                        if E_other == E:
                            for m in V.DOF_range[K_other]:
                                psi = Psi[m]
                                k = psi.k
                                H = 1
                                i_index.append(m)
                                j_index.append(n)
                                S = Sigma_local(phi, psi, E, k, H, d_2) + Sigma_nonlocal(phi, psi, E, E, k, H, d_2, Np=Np)
                                values.append(S)
                        else:
                            for m in V.DOF_range[K_other]:
                                psi = Psi[m]
                                k = psi.k
                                H = 1
                                i_index.append(m)
                                j_index.append(n)
                                values.append(Sigma_nonlocal(phi, psi, E, E_other, k, H, d_2, Np=Np))
                        
    values = np.array(values)
    i_index = np.array(i_index)
    j_index = np.array(j_index)
    
    
    A = coo_matrix( (values, (i_index, j_index)), shape=(N_DOF,N_DOF))
    A = csr_matrix(A)

    if full_matrix:
        A = A.toarray()

    return A

def exact_RHS(psi, E, k, H, d_2, t):
    d = psi.d
    d_y = d[1]
    N = E.N
    M = E.M
    l = E.l


    beta = sqrt(complex(k**2 - (t*pi/H)**2))

    F = 1j*k*l*exp(1j*beta*M[0])*exp(-1j*k*dot(d,M))*(dot(d,N) - d_2)*(exp( 1j*pi*t/H*M[1])*sinc(t*l/(2*H) - k*l*d_y/(2*pi)) + 
                                                                       exp(-1j*pi*t/H*M[1])*sinc(t*l/(2*H) + k*l*d_y/(2*pi)))

    if t == 0:
        S = 2j*k*l*dot(d,N)*d_2*exp(1j*(beta*M[0]-k*dot(d,M)))*sinc(k*l/(2*pi)*d[1])

    else:
        S = 1j*k*l*dot(d,N)*d_2*exp(1j*(beta*M[0]-k*dot(d,M)))*( k/conj(sqrt(complex(k**2 - (t*pi/H)**2))) *
                                                                        (exp( 1j*t*pi*M[1]/H)*sinc(t*l/(2*H) - k*l*d[1]/(2*pi)) + 
                                                                         exp(-1j*t*pi*M[1]/H)*sinc(t*l/(2*H) + k*l*d[1]/(2*pi))))
 
    return F + S




def AssembleRHS(V, Edges, k, H, d_2, t=0, full_sides=False):
    N_DOF = V.N_DOF
    b = np.zeros((N_DOF), dtype=np.complex128)
    Psi = V.TestFunctions

    for E in Edges:
        match E.Type:                
            case EdgeType.SIGMA_L:
                K = E.Triangles[0]
                for m in V.DOF_range[K]:
                    psi = Psi[m]
                    b[m] += exact_RHS(psi, E, k, H, d_2, t=t)
            case EdgeType.SIGMA_R:
                pass
    return b

