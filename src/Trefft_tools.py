
import numpy as np
from numpy import dot, pi, exp, sqrt, abs, conj
from numpy.lib.scimath import sqrt
from numpy.linalg import norm
from collections import namedtuple
from labels import EdgeType
from scipy.sparse import coo_matrix, csr_matrix, spmatrix, bsr_array
from geometry_tools import Edge
from numpy import sinc, cos
from exact_solutions import GreenFunctionImages, GreenFunctionModes
from integrators import fekete3 as int2D
from integrators import vec_fekete3 as int2D_vec

from domains import ScattererType


import numpy.typing as npt


real_array = npt.NDArray[np.floating]
complex_array = npt.NDArray[np.complexfloating]




TestFunction = namedtuple("TestFunction", ["n", "d"])



class TrefftzSpace:
    '''Defines a finite dimensional Trefftz space given
    a mesh, the number of plane-waves per element and 
    the wave-numbers.

    It can create test and trial functions, aswell as 
    actual functions.
    '''

    def __init__( self, Domain, DOF_per_element : tuple[int], kappa : float, n :dict[str, float | complex], th0=0 ):
        Omega = Domain.Omega
        self.Omega = Omega
        self.absorbing = Domain.scatterer_type == ScattererType.ABSORBING
        self.ScattererTriangles = Domain.ScattererTriangles
        self.N_trig = len(Omega.faces)
        self.kappa = kappa
        self.n = np.zeros(self.N_trig, dtype=np.array(list(n.values())).dtype)
        
        self._elements = list(Omega.Elements())
        for e in Omega.Elements():
            self.n[e.faces[0].nr] = n[e.mat]
        
        if hasattr(DOF_per_element, '__iter__'):
            assert self.N_trig == len(DOF_per_element)
            self.local_N_DOF = np.array( DOF_per_element )
        else:
            self.local_N_DOF = np.full(self.N_trig, fill_value=DOF_per_element, dtype=np.int32)

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
        return [ TestFunction( n= self.n[self.DOF_ownership[ID]], d=self.d[self.DOF_ownership[ID]][self.global_to_local[ID]]) for ID in range(self.N_DOF)]
    @property
    def TrialFunctions( self ):
        return [ TestFunction( n= self.n[self.DOF_ownership[ID]], d=self.d[self.DOF_ownership[ID]][self.global_to_local[ID]]) for ID in range(self.N_DOF)]




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
        
        k = np.sqrt(self.V.n[e]) * self.V.kappa
        P = self.DOFs[self.V.DOF_range[e]]
        D = self.V.d[e]

        r = np.array([x,y])
        y = sum( p*np.exp(1j*k*dot(d,r)) for (p,d) in zip(P,D) )
        return y 

    # def grad(self, x, y ):  !CLEAN THIS
    #     e = self.Element(x,y)
    #     if e < 0: # (x,y) outside the mesh
    #         return np.nan
    #     def grad(v):
    #         k = self.V.kappa[e]
    #         P = self.DOFs[self.V.DOF_range[e]]
    #         D = self.V.d[e]

    #         r = np.array([x,y])
    #         y = sum( 1j*k**p*np.exp(1j*k*dot(d,r))*dot(d,v) for (p,d) in zip(P,D) )
    #     return grad 
    def dx(self, x, y ):
        e = self.Element(x,y)
        if e < 0: # (x,y) outside the mesh
            return np.nan

        k = np.sqrt(self.V.n[e]) * self.V.kappa
        P = self.DOFs[self.V.DOF_range[e]]
        D = self.V.d[e]

        r = np.array([x,y])
        dudx = sum( 1j*k*p*np.exp(1j*k*dot(d,r))*d[0] for (p,d) in zip(P,D) )
        return dudx 

    def dy(self, x, y ):
        e = self.Element(x,y)
        if e < 0: # (x,y) outside the mesh
            return np.nan

        k = np.sqrt(self.V.n[e]) * self.V.kappa
        P = self.DOFs[self.V.DOF_range[e]]
        D = self.V.d[e]

        r = np.array([x,y])
        dudy = sum( 1j*k*p*np.exp(1j*k*dot(d,r))*d[1] for (p,d) in zip(P,D) )
        return dudy 


def Gamma_term(phi, psi, k, edge, d_1):

    d_m = psi.d
    d_n = phi.d
    
    M = edge.M
    l = edge.l
    N = edge.N
    T = edge.T

    I = -1j*k*l*(1 + d_1 * dot(d_n, N))*dot(d_m, N)*exp(1j*k*dot(d_n - d_m,M))*sinc(k*l/(2*pi)*dot(d_n-d_m,T))
    return I

def Gamma_local(k : complex, l : float, M : real_array, T : real_array, N : real_array,
                 d : real_array, d_d : real_array, d_1 : np.floating) -> complex_array:
    I = -1j*k*l*dot(d, N)[:,np.newaxis]*exp(1j*k*dot(d_d,M))*sinc(k*l/(2*pi)*dot(d_d,T))*(1 + d_1*dot(d, N))
    return I

# def Inner_local(k : complex, l : float, M : real_array, T : real_array, N : real_array,
#                  d : real_array, d_d : real_array, a : np.floating, b : np.floating) -> complex_array:
#     I = -1j*k*l*( np.add.outer(dot(d, N),dot(d, N))/2 + a + b*np.outer(dot(d, N),dot(d, N)))*exp(1j*k*dot(d_d,M))*sinc(k*l/(2*pi)*dot(d_d,T))
#     return I

def Inner_general_local(k : complex, l : float, M : real_array, T : real_array, N : real_array, n_n, n_m, 
                        d : real_array, a : np.floating, b : np.floating) -> complex_array:

    
    I = -1j*k*l*(a + np.add.outer(sqrt(n_m)*dot(d,N),sqrt(n_n)*dot(d,N))/2 + b*np.outer(sqrt(n_m)*dot(d,N),sqrt(n_n)*dot(d,N))) \
    *exp(-1j*k*np.subtract.outer(sqrt(n_m)*dot(d,M),sqrt(n_n)*dot(d,M)))                                             \
    *sinc(l*k/(2*pi)*np.subtract.outer(sqrt(n_m)*dot(d,T),sqrt(n_n)*dot(d,T)))

    return I


def Sigma_block_local(k : complex, l : float, M : real_array, T : real_array, N : real_array,
                 d : real_array, d_d : real_array, d_2 : np.floating) -> complex_array:
    I = -1j*k*l*(d_2 + dot(d,N))*exp(1j*k*dot(d_d,M))*sinc(k*l/(2*pi)*dot(d_d,T))
    return I





def Sigma_crossblock(k : complex, H : float, l_u : float, M_u : real_array, l_v : float, M_v : real_array, N : real_array,
                 d : real_array, d_2 : np.floating, N_DtN : int) -> complex_array:

    I1 = -1j*k*H*np.outer(dot(d,N),dot(d,N))*d_2*l_u/H*l_v/H*                                                                                      \
        exp(1j*k*np.subtract.outer(dot(d,M_u),dot(d,M_v)).transpose())*( np.outer(sinc(k*l_v/(2*pi)*d[:,1]),sinc(k*l_u/(2*pi)*d[:,1])) +           \
        1/2*sum( [ k**2 / abs(sqrt(complex(k**2 - (s*pi/H)**2)))**2 *                                                                              \
        np.outer( (exp(-1j*s*pi/H*M_v[1])*sinc(k*l_v/(2*pi)*d[:,1] + s*l_v/(2*H)) + exp( 1j*s*pi/H*M_v[1])*sinc(k*l_v/(2*pi)*d[:,1] - s*l_v/(2*H)) ),
                  (exp( 1j*s*pi/H*M_u[1])*sinc(k*l_u/(2*pi)*d[:,1] + s*l_u/(2*H)) + exp(-1j*s*pi/H*M_u[1])*sinc(k*l_u/(2*pi)*d[:,1] - s*l_u/(2*H)) ) )      
        for s in range(1, N_DtN)]) )


    I2 = -1j*k*H*np.outer((dot(d,N)-d_2),dot(d,N))*l_u/H*l_v/H*                                                                                      \
        exp(1j*k*np.subtract.outer(dot(d,M_u),dot(d,M_v)).transpose())*( np.outer(sinc(k*l_v/(2*pi)*d[:,1]),sinc(k*l_u/(2*pi)*d[:,1])) +             \
        1/2*sum( [ k / sqrt(complex(k**2 - (s*pi/H)**2)) *                                                                                           \
        np.outer ( ( exp(-1j*s*pi/H*M_v[1])*sinc(k*l_v/(2*pi)*d[:,1] + s*l_v/(2*H)) + exp( 1j*s*pi/H*M_v[1])*sinc(k*l_v/(2*pi)*d[:,1] - s*l_v/(2*H)) ) ,
                   ( exp( 1j*s*pi/H*M_u[1])*sinc(k*l_u/(2*pi)*d[:,1] + s*l_u/(2*H)) + exp(-1j*s*pi/H*M_u[1])*sinc(k*l_u/(2*pi)*d[:,1] - s*l_u/(2*H)) ) )
        for s in range(1,N_DtN)]) )

    I3 = 1j*k*H*dot(d,N)[:,np.newaxis]*d_2*l_u/H*l_v/H*                                                                                      \
        exp(1j*k*np.subtract.outer(dot(d,M_u),dot(d,M_v)).transpose())*( np.outer(sinc(k*l_v/(2*pi)*d[:,1]),sinc(k*l_u/(2*pi)*d[:,1])) +             \
        1/2*sum( [ k / conj(sqrt(complex(k**2 - (s*pi/H)**2))) *                                                                                           \
        np.outer ( ( exp(-1j*s*pi/H*M_v[1])*sinc(k*l_v/(2*pi)*d[:,1] + s*l_v/(2*H)) + exp( 1j*s*pi/H*M_v[1])*sinc(k*l_v/(2*pi)*d[:,1] - s*l_v/(2*H)) ) ,
                   ( exp( 1j*s*pi/H*M_u[1])*sinc(k*l_u/(2*pi)*d[:,1] + s*l_u/(2*H)) + exp(-1j*s*pi/H*M_u[1])*sinc(k*l_u/(2*pi)*d[:,1] - s*l_u/(2*H)) ) )
        for s in range(1,N_DtN)]) )

 
    return I1 + I2 + I3


def SoundSoft_local(k : complex, l : float, M : real_array, T : real_array, N : real_array,
                 d : real_array, d_d : real_array, a : np.floating) -> complex_array:
    
    I = -1j*k*l*exp(1j*k*dot(d_d, M))*sinc(k*l/(2*pi)*dot(d_d,T))*(dot(d, N) + a)
    return  I




# bsr works block by block. indprt must allways have N+1 items where N is the number of block rows
# indices and data must have the same number of items, as they locate the blocks for each "block row"


# This assumes they are sorted, i.e. [e.Triangles[0]] appears sorted 
def Gamma_global(k : complex, N_elems : int,  Edges : real_array,
                 d : real_array, d_d : real_array, d_1 : np.floating, l_max = None) -> complex_array:
    N_p = d_d.shape[0]
    N_wall_sides = len(Edges)
    data = np.zeros((N_wall_sides, N_p, N_p), dtype=np.complex128)
    indices = np.array([e.Triangles[0] for e in Edges])
    indptr =  np.concatenate([ np.zeros(indices[0]+1, dtype=np.int32), 
                               np.arange(1,len(indices)).repeat(indices[1:] - indices[:-1]), 
                               np.full(N_elems - indices[-1], len(indices))])

    if l_max is None:
        for (i, edge) in enumerate(Edges):
            data[i,:,:] = Gamma_local( k=k, l=edge.l, M=edge.M, T=edge.T, N=edge.N, d=d, d_d=d_d, d_1=d_1 )
    else:
        for (i, edge) in enumerate(Edges):
            #factor = (1+l_max/edge.l)
            factor = l_max / edge.l
            data[i,:,:] = Gamma_local( k=k, l=edge.l, M=edge.M, T=edge.T, N=edge.N, d=d, d_d=d_d, d_1=d_1*factor )
    
    A = bsr_array((data, indices, indptr), shape=(N_elems*N_p, N_elems*N_p))    
    return A



def Inner_PP_global(k : complex, N_elems : int,  Edges : real_array,
                 d : real_array, d_d : real_array, n : complex_array, a : np.floating, b : np.floating, l_max  = None) -> complex_array:
    N_p = d_d.shape[0]
    N_inner_sides = len(Edges)
    data = np.zeros((N_inner_sides, N_p, N_p), dtype=np.complex128)
    indices = np.array([e.Triangles[0] for e in Edges])
    indptr =  np.concatenate([ np.zeros(indices[0]+1, dtype=np.int32), 
                               np.arange(1,len(indices)).repeat(indices[1:] - indices[:-1]), 
                               np.full(N_elems - indices[-1], len(indices))])


    if l_max is None:
        for (i, edge) in enumerate(Edges):
            data[i,:,:] = Inner_general_local(k = k, l = edge.l, M = edge.M, T = edge.T, N = edge.N, n_m=n[i], n_n=n[i], d = d, a = a, b = b)
    else:
        for (i, edge) in enumerate(Edges):
            #factor = (1+l_max/edge.l)
            factor = l_max / edge.l
            data[i,:,:] = Inner_general_local(k = k, l = edge.l, M = edge.M, T = edge.T, N = edge.N, n_m=n[i], n_n=n[i], d = d,
                                            a = a*factor, b = b*factor)

    A = bsr_array((data, indices, indptr), shape=(N_elems*N_p, N_elems*N_p))
    return A

def Inner_MM_global(k : complex, N_elems : int,  Edges : real_array,
                 d : real_array, d_d : real_array, n : complex_array, a : np.floating, b : np.floating, l_max = None) -> complex_array:
    N_p = d_d.shape[0]
    N_inner_sides = len(Edges)
    data = np.zeros((N_inner_sides, N_p, N_p), dtype=np.complex128)
    indices = np.array([e.Triangles[1] for e in Edges])
    indptr =  np.concatenate([ np.zeros(indices[0]+1, dtype=np.int32), 
                               np.arange(1,len(indices)).repeat(indices[1:] - indices[:-1]), 
                               np.full(N_elems - indices[-1], len(indices))])


    if l_max is None:
        for (i, edge) in enumerate(Edges):
            data[i,:,:] = -Inner_general_local( k=k, l=edge.l, M=edge.M, T=edge.T, N=edge.N, n_m=n[i], n_n=n[i], d=d, a=-a, b=-b)
    else:
        for (i, edge) in enumerate(Edges):
            #factor = (1+l_max/edge.l)
            factor = l_max / edge.l
            data[i,:,:] = -Inner_general_local( k=k, l=edge.l, M=edge.M, T=edge.T, N=edge.N, n_m=n[i], n_n=n[i], d=d,
                                            a = -a*factor, b = -b*factor)
    A = bsr_array((data, indices, indptr), shape=(N_elems*N_p, N_elems*N_p))
    return A


# plus for m, minus for n
def Inner_PM_global(k : complex, N_elems : int,  Edges : real_array,
                 d : real_array, d_d : real_array, n_n : complex_array, n_m : complex_array,
                 a : np.floating, b : np.floating, l_max  = None) -> complex_array:
    N_p = d_d.shape[0]
    N_inner_sides = len(Edges)
    data = np.zeros((N_inner_sides, N_p, N_p), dtype=np.complex128)
    indices_M = np.array([e.Triangles[1] for e in Edges])
    indices_P = np.array([e.Triangles[0] for e in Edges])
    indptr =  np.concatenate([ np.zeros(indices_P[0]+1, dtype=np.int32), 
                                np.arange(1,len(indices_P)).repeat(indices_P[1:] - indices_P[:-1]), 
                                np.full(shape = N_elems - indices_P[-1], fill_value = len(indices_P))])

    #  [ ., ., ., 3, 4, ., 6, ., 8, 8, ., .] -> [ 0, 0, 0, 0, 1, 2, 2, 3, 3, 5, 5, 5 ]
    if l_max is None:
        for (i, edge) in enumerate(Edges):
            data[i,:,:] = Inner_general_local( k=k, l=edge.l, M=edge.M, T=edge.T, N=edge.N, d=d, n_m=n_m[i], n_n=n_n[i], a=-a, b=-b)
    else:
        for (i, edge) in enumerate(Edges):
            #factor = (1+l_max/edge.l)
            factor = l_max / edge.l
            data[i,:,:] = Inner_general_local( k=k, l=edge.l, M=edge.M, T=edge.T, N=edge.N, d=d, n_m=n_m[i], n_n=n_n[i],
                                              a = -a*factor, b = -b*factor)

    A = bsr_array((data, indices_M, indptr), shape=(N_elems*N_p, N_elems*N_p))
    return A

def Inner_MP_global(k : complex, N_elems : int, Edges : real_array,
                 d : real_array, d_d : real_array, n_n : complex_array, n_m : complex_array,
                 a : np.floating, b : np.floating, l_max  = None) -> complex_array:
    N_p = d_d.shape[0]
    N_inner_sides = len(Edges)
    data = np.zeros((N_inner_sides, N_p, N_p), dtype=np.complex128)
    indices_M = np.array([e.Triangles[0] for e in Edges])
    indices_P = np.array([e.Triangles[1] for e in Edges])
    indptr =  np.concatenate([ np.zeros(indices_P[0]+1, dtype=np.int32), 
                                np.arange(1,len(indices_P)).repeat(indices_P[1:] - indices_P[:-1]), 
                                np.full(shape = N_elems - indices_P[-1], fill_value = len(indices_P))])
    if l_max is None:
        for (i, edge) in enumerate(Edges):
            data[i,:,:] = -Inner_general_local( k=k, l=edge.l, M=edge.M, T=edge.T, N=edge.N, d=d, n_m=n_m[i], n_n=n_n[i], a=a, b=b)
    else:
        for (i, edge) in enumerate(Edges):
            #factor = (1+l_max/edge.l)
            factor = l_max / edge.l
            data[i,:,:] = -Inner_general_local( k=k, l=edge.l, M=edge.M, T=edge.T, N=edge.N, d=d, n_m=n_m[i], n_n=n_n[i],
                                               a=a*factor, b=b*factor)
    
    A = bsr_array((data, indices_M, indptr), shape=(N_elems*N_p, N_elems*N_p))
    return A







def Sigma_global(k : complex, N_elems : int,  Edges : real_array,
                 d : real_array, d_d : real_array, d_2 : np.floating) -> complex_array:
    N_p = d_d.shape[0]
    N_sigma_sides = len(Edges)
    data = np.zeros((N_sigma_sides, N_p, N_p), dtype=np.complex128)
    indices = np.array([e.Triangles[0] for e in Edges])
    indptr =  np.concatenate([ np.zeros(indices[0]+1, dtype=np.int32), 
                               np.arange(1,len(indices)).repeat(indices[1:] - indices[:-1]), 
                               np.full(N_elems - indices[-1], len(indices))])


    for (i, edge) in enumerate(Edges):
        data[i,:,:] = Sigma_block_local( k=k, l=edge.l, M=edge.M, T=edge.T, N=edge.N, d=d, d_d=d_d, d_2=d_2 )
    A = bsr_array((data, indices, indptr), shape=(N_elems*N_p, N_elems*N_p))    
    return A

def Sigma_cross_global(k : complex, H : float,  N_elems : int,  Edges : real_array,
                 d : real_array, N : real_array, d_2 : np.floating, N_DtN : int) -> complex_array:
    N_p = d.shape[0]
    N_sigma_sides = len(Edges)
    data = np.zeros((N_sigma_sides**2, N_p, N_p), dtype=np.complex128)
    ind_rows = np.array([e.Triangles[0] for e in Edges])

    indices = np.tile( ind_rows, N_sigma_sides)

    
    indptr =  np.concatenate([ np.zeros(ind_rows[0]+1, dtype=np.int32), 
                               np.arange(1,len(ind_rows)).repeat(ind_rows[1:] - ind_rows[:-1]), 
                               np.full(N_elems - ind_rows[-1], len(ind_rows))])*N_sigma_sides

    # [ ., ., 2, 3, ., 4, ., . ] -> [ 0, 0, 0, 3, 6, 6, 9, 9, 9]
    # [ ., ., 2, 3, ., 4, ., . ] -> [ 0, 0, 0, 1, 2, 2, 3, 3, 3] * 3

    for (i, edge_v) in enumerate(Edges):
        for (j, edge_u) in enumerate(Edges):
            ij = N_sigma_sides*i + j
            data[ij,:,:] = Sigma_crossblock(k = k, H = H, l_u = edge_u.l, M_u = edge_u.M, l_v = edge_v.l, M_v = edge_v.M, N = N,
                                            d = d, d_2 = d_2, N_DtN = N_DtN)
    A = bsr_array((data, indices, indptr), shape=(N_elems*N_p, N_elems*N_p))    
    return A




def SoundSoft_global(k : complex, N_elems : int,  Edges : real_array,
                 d : real_array, d_d : real_array, a : np.floating) -> complex_array:
    N_p = d_d.shape[0]
    N_dOmega_sides = len(Edges)
    data = np.zeros((N_dOmega_sides, N_p, N_p), dtype=np.complex128)
    indices = np.array([e.Triangles[0] for e in Edges])
    indptr =  np.concatenate([ np.zeros(indices[0]+1, dtype=np.int32), 
                               np.arange(1,len(indices)).repeat(indices[1:] - indices[:-1]), 
                               np.full(N_elems - indices[-1], len(indices))])


    for (i, edge) in enumerate(Edges):
        data[i,:,:] = SoundSoft_local( k=k, l=edge.l, M=edge.M, T=edge.T, N=edge.N, d=d, d_d=d_d, a=a )
    A = bsr_array((data, indices, indptr), shape=(N_elems*N_p, N_elems*N_p))    
    return A



def absorption_term_local(r_A, r_B, r_C, d_d, n, k):
    k_i =  k * sqrt(n)
    def f(x,y):
        d_dx = np.expand_dims( d_d[:,:,0], axis=-1)*x.reshape((1,1,-1))
        d_dy = np.expand_dims( d_d[:,:,1], axis=-1)*y.reshape((1,1,-1))
        return np.exp(1j*k_i*(d_dx + d_dy))

    I = -2*1j*k**2*np.imag(n)*int2D_vec( f, r_A=r_A, r_B=r_B, r_C=r_C)

    return I

def absorption_term_global(Triangles : list, N_elems : int, d_d : real_array, n : list, k : float ):
    N_p = d_d.shape[0]
    N_Triangles = len(Triangles)
    data = np.zeros((N_Triangles, N_p, N_p), dtype=np.complex128)
    indices = np.array([T.index for T in Triangles])
    indptr =  np.concatenate([ np.zeros(indices[0]+1, dtype=np.int32), 
                               np.arange(1,len(indices)).repeat(indices[1:] - indices[:-1]), 
                               np.full(N_elems - indices[-1], len(indices))])

    for (i, T) in enumerate(Triangles):
        data[i,:,:] = absorption_term_local(r_A=T.A, r_B=T.B, r_C=T.C, d_d = d_d, n = n[i], k = k)
    A = bsr_array((data, indices, indptr), shape=(N_elems*N_p, N_elems*N_p))    
    return A



def Inner_term_general(phi, psi, edge, k, a, b):

    d_m = psi.d
    d_n = phi.d

    k_n = k * sqrt(phi.n)
    k_m = k * sqrt(psi.n)

    
    M = edge.M
    N = edge.N
    T = edge.T
    l = edge.l

    I = -1j*l/2*(2*a*k + k_n*dot(d_n,N) + k_m*dot(d_m,N) + 2*b/k*k_n*dot(d_n,N)*k_m*dot(d_m,N))*exp(1j*dot(k_n*d_n - k_m*d_m,M))*sinc(l/(2*pi)*dot(k_n*d_n - k_m*d_m,T))

    return I







def sound_soft_term(phi, psi, k, edge, a):

    d_m = psi.d
    d_n = phi.d
    
    M = edge.M
    N = edge.N
    T = edge.T
    l = edge.l

    I = -1j*k*l*(dot(d_n, N) + a)* exp(1j*k*dot(d_n - d_m, M))*sinc(k*l/(2*pi)*dot(d_n-d_m,T))

    return  I


def Sigma_local(phi, psi, k, edge, d_2):

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

    I1 = -1j*k*H*dot(d_n,N)*dot(d_m,N)*d_2*exp(1j*k*(dot(d_n,M_u) - dot(d_m,M_v)))*l_u/H*l_v/H*(
        sinc(k*l_u/(2*pi)*d_n[1])*sinc(k*l_v/(2*pi)*d_m[1]) + 1/2*sum( [ k**2 / abs(sqrt(complex(k**2 - (s*pi/H)**2)))**2 * (
        exp( 1j*s*pi/H*M_u[1])*sinc(k*l_u/(2*pi)*d_n[1] + s*l_u/(2*H)) + exp(-1j*s*pi/H*M_u[1])*sinc(k*l_u/(2*pi)*d_n[1] - s*l_u/(2*H)) ) *(
        exp(-1j*s*pi/H*M_v[1])*sinc(k*l_v/(2*pi)*d_m[1] + s*l_v/(2*H)) + exp( 1j*s*pi/H*M_v[1])*sinc(k*l_v/(2*pi)*d_m[1] - s*l_v/(2*H)) )
        for s in range(1,Np)]) )
    
    I2 = -1j*k*H*dot(d_n,N)*(dot(d_m,N)-d_2)*exp(1j*k*(dot(d_n,M_u) - dot(d_m,M_v)))*l_u/H*l_v/H*(
        sinc(k*l_u/(2*pi)*d_n[1])*sinc(k*l_v/(2*pi)*d_m[1]) + 1/2*sum( [ k / sqrt(complex(k**2 - (s*pi/H)**2)) * (
        exp( 1j*s*pi/H*M_u[1])*sinc(k*l_u/(2*pi)*d_n[1] + s*l_u/(2*H)) + exp(-1j*s*pi/H*M_u[1])*sinc(k*l_u/(2*pi)*d_n[1] - s*l_u/(2*H)) ) *(
        exp(-1j*s*pi/H*M_v[1])*sinc(k*l_v/(2*pi)*d_m[1] + s*l_v/(2*H)) + exp( 1j*s*pi/H*M_v[1])*sinc(k*l_v/(2*pi)*d_m[1] - s*l_v/(2*H)) )
        for s in range(1,Np)]) )
    
    I3 = 1j*k*H*dot(d_m,N)*d_2*exp(1j*k*(dot(d_n,M_u) - dot(d_m,M_v)))*l_u/H*l_v/H*(
        sinc(k*l_u/(2*pi)*d_n[1])*sinc(k*l_v/(2*pi)*d_m[1]) + 1/2*sum( [ k / conj(sqrt(complex(k**2 - (s*pi/H)**2))) * (
        exp( 1j*s*pi/H*M_u[1])*sinc(k*l_u/(2*pi)*d_n[1] + s*l_u/(2*H)) + exp(-1j*s*pi/H*M_u[1])*sinc(k*l_u/(2*pi)*d_n[1] - s*l_u/(2*H)) ) *(
        exp(-1j*s*pi/H*M_v[1])*sinc(k*l_v/(2*pi)*d_m[1] + s*l_v/(2*H)) + exp( 1j*s*pi/H*M_v[1])*sinc(k*l_v/(2*pi)*d_m[1] - s*l_v/(2*H)) )
        for s in range(1,Np)]) )

    return  I1 + I2 + I3


def absorption_term( phi, psi, r_A, r_B, r_C, k):
    n = phi.n
    k_i =  k * sqrt(n)
    d_n = phi.d
    d_m = psi.d
    I = -2*1j*k**2*np.imag(n)*int2D(lambda x, y : np.exp(1j*k_i*dot(d_n - d_m,np.array((x,y)))), r_A=r_A, r_B=r_B, r_C=r_C)

    return I
    


def AssembleMatrix(V : TrefftzSpace,  Edges : tuple[Edge], 
                   H : float, a = 0.5, b = 0.5, d_1 = 0.5, d_2 = 0.5, 
                   Np=10, full_matrix = False) -> spmatrix:
    '''Assembles de matrix for the bilinear form.
    a, b, d_1 and d_2 are the coefficients of the regularizing terms.
    Use full_matrix = Truee if the returned matrix should NOT be a sparse
    matrix.'''


    N_DOF = V.N_DOF

    values = []
    i_index = []
    j_index = []


    Side_edges  : dict[EdgeType, list] = { EdgeType.SIGMA_L : [], EdgeType.SIGMA_R : []} 
    
    for E in Edges:
        match E.Type:
            case EdgeType.SIGMA_L | EdgeType.SIGMA_R:
                Side_edges[E.Type].append(E)
            case _:
                pass

    N_Edges = len(Edges)

    
    # filling the vectors
    if np.isscalar(a):
        a_vec = np.full(N_Edges,a)
    else:
        a_vec = a 
        
    if np.isscalar(b):
        b_vec = np.full(N_Edges,b)
    else:
        b_vec = b 
    
    if np.isscalar(d_1):
        d_1_vec = np.full(N_Edges,d_1)
    else:
        d_1_vec = d_1 
    

    if np.isscalar(d_2):
        d_2_vec = np.full(N_Edges,d_2)
    else:
        d_2_vec = d_2
    


    Phi = V.TrialFunctions
    Psi = V.TestFunctions # currently the same spaces 

    k = V.kappa

    for (E, a, b, d_1, d_2) in zip(Edges, a_vec, b_vec, d_1_vec, d_2_vec):
        match E.Type:
            case EdgeType.INNER:
                K_plus, K_minus = E.Triangles

                for n in V.DOF_range[K_plus]:
                    phi = Phi[n]
                    for m in V.DOF_range[K_plus]:
                        psi = Psi[m]
                        i_index.append(m)
                        j_index.append(n)
                        values.append(Inner_term_general(phi, psi, E, k, a, b))

                for n in V.DOF_range[K_minus]:
                    phi = Phi[n]
                    for m in V.DOF_range[K_plus]:
                        psi = Psi[m]
                        i_index.append(m)
                        j_index.append(n)
                        values.append(Inner_term_general(phi, psi, E, k, -a, -b))


                for n in V.DOF_range[K_plus]:
                    phi = Phi[n]
                    for m in V.DOF_range[K_minus]:
                        psi = Psi[m]
                        i_index.append(m)
                        j_index.append(n)
                        values.append(-Inner_term_general(phi, psi, E, k, a, b))

                for n in V.DOF_range[K_minus]:
                    phi = Phi[n]
                    for m in V.DOF_range[K_minus]:
                        psi = Psi[m]
                        i_index.append(m)
                        j_index.append(n)
                        values.append(-Inner_term_general(phi, psi, E, k, -a, -b))


            case EdgeType.GAMMA:
                K = E.Triangles[0]
                for m in V.DOF_range[K]:
                    psi = Psi[m]
                    for n in V.DOF_range[K]:
                        phi = Phi[n]
                        i_index.append(m)
                        j_index.append(n)
                        values.append(Gamma_term(phi, psi, k, E, d_1))
                    

            case EdgeType.D_OMEGA | EdgeType.COVER:
                K = E.Triangles[0]
                for m in V.DOF_range[K]:
                    psi = Psi[m]
                    for n in V.DOF_range[K]:
                        phi = Phi[n]
                        i_index.append(m)
                        j_index.append(n)
                        values.append(sound_soft_term(phi, psi, k, E, a))


            case EdgeType.SIGMA_L | EdgeType.SIGMA_R:
                K = E.Triangles[0]
                for n in V.DOF_range[K]:
                    phi = Phi[n]
                    for E_other in Side_edges[E.Type]:
                        K_other = E_other.Triangles[0]
                        if E_other == E:
                            for m in V.DOF_range[K_other]:
                                psi = Psi[m]
                                i_index.append(m)
                                j_index.append(n)
                                S = Sigma_local(phi, psi, k, E, d_2) + Sigma_nonlocal(phi, psi, E, E, k, H, d_2, Np=Np)
                                values.append(S)
                        else:
                            for m in V.DOF_range[K_other]:
                                psi = Psi[m]
                                i_index.append(m)
                                j_index.append(n)
                                values.append(Sigma_nonlocal(phi, psi, E, E_other, k, H, d_2, Np=Np))
    if V.absorbing:                    
        for T in V.ScattererTriangles:
            r_A, r_B, r_C = T.A, T.B, T.C
            T_index = T.index
            for n in V.DOF_range[T_index]:
                phi = Phi[n]
                for m in V.DOF_range[T_index]:
                    psi = Psi[m]
                    i_index.append(m)
                    j_index.append(n)
                    values.append(absorption_term( phi, psi, r_A, r_B, r_C, k))

          
    A = coo_matrix( (values, (i_index, j_index)), shape=(N_DOF,N_DOF))
    A = csr_matrix(A)

    if full_matrix:
        A = A.toarray()

    return A



def mode_RHS(psi, E, k, H, d_2, t):
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



def AssembleRHS(V, Edges, k, H, d_2, t=0):
    N_DOF = V.N_DOF
    b = np.zeros((N_DOF), dtype=np.complex128)
    Psi = V.TestFunctions
    N_Edges = len(Edges)
    if np.isscalar(d_2):
        d_2_vec = np.full(N_Edges,d_2)
    else:
        d_2_vec = d_2


    for (E, d_2) in zip(Edges,d_2_vec):
        match E.Type:                
            case EdgeType.SIGMA_L:
                K = E.Triangles[0]
                for m in V.DOF_range[K]:
                    psi = Psi[m]
                    b[m] += mode_RHS(psi, E, k, H, d_2, t=t)
            case EdgeType.SIGMA_R:
                pass
    return b



# def Green_RHS(psi, E, k, H, a, x_0, y_0, modes=False, n_modes=20):
#     M = E.M
#     T = E.T 
#     N = E.N
#     l = E.l

#     d_m = psi.d

#     Npoints = 200
#     t = np.linspace(-l/2,l/2,Npoints)
#     if modes:
#         g = GreenFunctionModes(k, H, M + np.outer(t,T), x_0, y_0, M=n_modes)    
#     else:
#         g = GreenFunctionImages(k, H, M + np.outer(t,T), x_0, y_0, M=n_modes)
#     I = -1j*k*( dot(d_m,N) - a)* exp(-1j*k*dot(d_m, M)) * Int( -g*exp(-1j*k*dot(d_m, T)*t), t)
#     return I




# def AssembleGreenRHS(V, Edges, k, H, a, x_0 = 0., y_0=0.5, modes=True, M=20):
#     N_DOF = V.N_DOF
#     b = np.zeros((N_DOF), dtype=np.complex128)
#     Psi = V.TestFunctions
#     N_edges = len(Edges)
#     if np.isscalar(a):
#         a_vec = np.full(N_edges,a)
#     else:
#         a_vec = a

#     for (E, a)  in zip(Edges,a_vec):
#         match E.Type:                
#             case EdgeType.D_OMEGA | EdgeType.COVER:
#                 K = E.Triangles[0]
#                 for m in V.DOF_range[K]:
#                     psi = Psi[m]
#                     b[m] += Green_RHS(psi, E, k, H, a, x_0, y_0, modes=modes, n_modes=M)
#     return b


def AssembleGreenRHS_left(V, Edges, k, H, d_2, x_0 = 0., y_0=0.5, M=20):
    N_DOF = V.N_DOF
    b = np.zeros((N_DOF), dtype=np.complex128)
    Psi = V.TestFunctions
    N_Edges = len(Edges)
    if np.isscalar(d_2):
        d_2_vec = np.full(N_Edges, d_2)
    else:
        d_2_vec = d_2


    for (E, d_2) in zip(Edges, d_2_vec):
        match E.Type:                
            case EdgeType.SIGMA_L:
                K = E.Triangles[0]
                for m in V.DOF_range[K]:
                    psi = Psi[m]

                    b[m] += -1/(2*1j*k*H)*exp(-1j*k*x_0)*mode_RHS(psi, E, k, H, d_2, t=0)

                    for t in range(1,M):
                        betaH = np.emath.sqrt( (k*H)**2 - (t*pi)**2)
                        b[m] += -1/(1j*betaH)*exp(-1j*betaH*x_0/H)*cos(t*pi*y_0/H)*mode_RHS(psi, E, k, H, d_2, t=t)

    return b


def Assemble_blockMatrix(V : TrefftzSpace,  Edges : tuple[Edge], th_0 : float, 
                   H : float, k=0.8, N_p = 3, a = 1/2,  b = 1/2, d_1 = 1/2, d_2=1/2, N_DtN = 15, rescaled_fluxes=False):


    wall_edges = []
    inner_edges = []
    sigma_edges = []
    sigma_edges_L = []
    sigma_edges_R = []
    d_Omega_edges = []
    for E in Edges:
        match E.Type:
            case EdgeType.GAMMA:
                wall_edges.append(E)
            case EdgeType.INNER:
                inner_edges.append(E)
            case EdgeType.SIGMA_L :
                sigma_edges.append(E)
                sigma_edges_L.append(E)
            case EdgeType.SIGMA_R :
                sigma_edges.append(E)
                sigma_edges_R.append(E)
            case EdgeType.D_OMEGA:
                d_Omega_edges.append(E)
            case _:
                pass
    if rescaled_fluxes:
        l_max = max([e.l for e in Edges])
    else:
        l_max = None

# this better for the numpy based mesh

    # if not isinstance(a,np.ndarray):
    #     a = np.full( shape=len(Edges), fill_value= a)

    # if not isinstance(b,np.ndarray):
    #     b = np.full( shape=len(Edges), fill_value= b)

    # if not isinstance(d_1,np.ndarray):
    #     d_1 = np.full( shape=len(Edges), fill_value= d_1)

    # if not isinstance(d_2,np.ndarray):
    #     d_2 = np.full( shape=len(Edges), fill_value= d_2)



    d_d = np.zeros( [N_p,N_p,2], dtype=np.float64)
    d = np.zeros( [N_p,2], dtype=np.float64)
    
    thetas = np.linspace(0,2*np.pi,N_p,endpoint=False) + th_0
    d_d[:,:,0] = - np.subtract.outer(np.cos(thetas), np.cos(thetas))
    d_d[:,:,1] = - np.subtract.outer(np.sin(thetas), np.sin(thetas))
    d[:,0] = np.cos(thetas)
    d[:,1] = np.sin(thetas)

    
    wall_edges.sort(key= lambda e : e.Triangles[0])
    A_block = Gamma_global(k=k, N_elems = V.N_trig, Edges = wall_edges, d=d, d_d=d_d, d_1=d_1, l_max=l_max )

    inner_edges.sort(key= lambda e : e.Triangles[0])
    n = [V.n[e.Triangles[0]] for e in inner_edges]
    A_block += Inner_PP_global(k=k, N_elems = V.N_trig, Edges = inner_edges, d=d, d_d=d_d, n=n, a=a, b=b, l_max=l_max)

    inner_edges.sort(key= lambda e : e.Triangles[1])
    n = [V.n[e.Triangles[1]] for e in inner_edges]
    A_block += Inner_MM_global(k=k, N_elems = V.N_trig, Edges = inner_edges, d=d, d_d=d_d, n=n,  a=a, b=b, l_max=l_max)

    inner_edges.sort(key= lambda e : e.Triangles[0])
    n_m = [V.n[e.Triangles[0]] for e in inner_edges]
    n_n = [V.n[e.Triangles[1]] for e in inner_edges]
    A_block += Inner_PM_global(k=k, N_elems = V.N_trig, Edges = inner_edges, d=d, d_d=d_d, n_m=n_m, n_n=n_n, a=a, b=b, l_max=l_max)

    inner_edges.sort(key= lambda e : e.Triangles[1])
    n_n = [V.n[e.Triangles[0]] for e in inner_edges]
    n_m = [V.n[e.Triangles[1]] for e in inner_edges]

    A_block += Inner_MP_global(k=k, N_elems = V.N_trig, Edges = inner_edges, d=d, d_d=d_d, n_m=n_m, n_n=n_n, a=a, b=b, l_max=l_max)




    sigma_edges.sort(key= lambda e : e.Triangles[0])
    A_block += Sigma_global(k=k, N_elems = V.N_trig,  Edges = sigma_edges, d=d, d_d=d_d, d_2=d_2 )

    sigma_edges_L.sort(key= lambda e : e.Triangles[0])
    A_block += Sigma_cross_global(k=k, H=H,  N_elems=V.N_trig,  Edges=sigma_edges_L, d=d,
                                   N = np.array([-1,0]), d_2=d_2, N_DtN=N_DtN)    

    sigma_edges_R.sort(key= lambda e : e.Triangles[0])
    A_block += Sigma_cross_global(k=k, H=H,  N_elems=V.N_trig,  Edges=sigma_edges_R, d=d,
                                   N = np.array([1,0]), d_2=d_2, N_DtN=N_DtN)    

    if len(d_Omega_edges) > 0:
        print('hola')
        d_Omega_edges.sort(key= lambda e : e.Triangles[0])
        A_block += SoundSoft_global(k=k, N_elems = V.N_trig, Edges = d_Omega_edges, d=d, d_d=d_d, a=a )


    if V.n.dtype == np.complexfloating:
        ScattererTriangles = V.ScattererTriangles
        ScattererTriangles.sort( key = lambda T  : T.index )
        n = [ V.n[T.index] for T in ScattererTriangles] 
        print('we are in the absorbing case')
        A_block += absorption_term_global( Triangles=ScattererTriangles, N_elems= V.N_trig, d_d=d_d, n=n, k=k) 

    
    return A_block 

