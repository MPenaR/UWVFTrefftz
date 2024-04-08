
import numpy as np
from numpy import dot, pi, exp, sqrt, sin, abs, conj
from numpy.linalg import norm
from collections import namedtuple
from labels import EdgeType
from scipy.sparse import coo_matrix, csr_matrix, spmatrix
from geometry_tools import Edge
from numpy import sinc, cos



TestFunction = namedtuple("TestFunction", ["k", "d"])



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




def Sigma_term(phi, psi, edge, d_2, Np = 15):

    d_n = phi.d
    d_m = psi.d
    k = phi.k

    d_mx = d_m[0]
    d_my = d_m[1]
    d_nx = d_n[0]
    d_ny = d_n[1]

    P = edge.P
    N = edge.N
    T = edge.T
    M = edge.M
    l = edge.l

    
    H = np.abs(P[1])
    
    x  = P[0]/H

    kH = k*H


    d_nN = dot(d_n,N)
    d_mN = dot(d_m,N)
    
    #first term
    #N(grad(u))*grad(v)
    F = -1j*k*l*exp(1j*(d_nx-d_mx)*kH*x)*d_mN*d_nN*( sinc(k*l/(2*pi)*d_ny)*sinc(k*l/(2*pi)*d_my) + 
         0.5*sum([kH/sqrt(complex(kH**2 - (s*pi)**2)) * (sinc(k*l/(2*pi)*d_ny + s) + sinc(k*l/(2*pi)*d_ny - s)) 
                                                      * (sinc(k*l/(2*pi)*d_my + s) + sinc(k*l/(2*pi)*d_my - s)) for s in range(1,Np)]) )                                           

    # grad(u)*v
    S1 = -1j*k*l*d_nN*exp(1j*k*dot(d_n-d_m,M))*sinc(k*l/(2*pi)*dot(d_n-d_m,T))



    # cross-terms
    # d2*N(grad(u))*v
    C1 = 1j*k*l*exp(1j*(d_nx-d_mx)*kH*x)*d_2*d_nN*( sinc(k*l/(2*pi)*d_ny)*sinc(k*l/(2*pi)*d_my) + 
         0.5*sum([kH/sqrt(complex(kH**2 - (s*pi)**2)) * (sinc(k*l/(2*pi)*d_ny + s) + sinc(k*l/(2*pi)*d_ny - s)) 
                                                      * (sinc(k*l/(2*pi)*d_my + s) + sinc(k*l/(2*pi)*d_my - s)) for s in range(1,Np)]) )                                           

    # d2*u*N(grad(v))
    C2 = 1j*k*l*exp(1j*(d_nx-d_mx)*kH*x)*d_2*d_mN*( sinc(k*l/(2*pi)*d_ny)*sinc(k*l/(2*pi)*d_my) + 
         0.5*sum([kH/conj(sqrt(complex(kH**2 - (s*pi)**2))) * (sinc(k*l/(2*pi)*d_ny + s) + sinc(k*l/(2*pi)*d_ny - s)) 
                                                            * (sinc(k*l/(2*pi)*d_my + s) + sinc(k*l/(2*pi)*d_my - s)) for s in range(1,Np)]) )                                           



    # d2*u*v
    S2 = -1j*k*l*d_2*exp(1j*k*dot(d_n-d_m,M))*sinc(k*l/(2*pi)*dot(d_n-d_m,T))


    #d2*N(grad(u))*N(grad(v)) 

    NN = -1j*k*l*exp(1j*(d_nx-d_mx)*kH*x)*d_2*d_mN*d_nN*( sinc(k*l/(2*pi)*d_ny)*sinc(k*l/(2*pi)*d_my) + 
         0.5*sum([kH**2/abs(sqrt(complex(kH**2 - (s*pi)**2)))**2 * (sinc(k*l/(2*pi)*d_ny + s) + sinc(k*l/(2*pi)*d_ny - s)) 
                                                                 * (sinc(k*l/(2*pi)*d_my + s) + sinc(k*l/(2*pi)*d_my - s)) for s in range(1,Np)]) )                                            

    return F + S1 + NN + C1 + C2 + S2





def Sigma_broken(phi, psi, edge, k, H, d_2, Np = 15):

    d_n = phi.d
    d_m = psi.d

    d_mx = d_m[0]
    d_my = d_m[1]
    d_nx = d_n[0]
    d_ny = d_n[1]
    


    kH = k*H
    
    x = edge.P[0]
    P_y = edge.P[1]
    Q_y = edge.Q[1]
     


    N = edge.N

    d_nN = dot(d_n,N)
    d_mN = dot(d_m,N)

    #CENTRED FLUXES
    
    #first terms
    I1 = -1j*kH*d_mN*d_nN*exp(1j*(d_nx-d_mx)*kH*x/H)

    if np.isclose(d_ny, 0, 1E-3) and np.isclose(d_my, 0, 1E-3):
        F = I1 *( Q_y - P_y) / H
    elif np.isclose(d_ny, 0, 1E-3):
        F = I1 * ( exp(-1j*kH*d_my*Q_y/H) - exp(-1j*kH*d_my*P_y/H)) / (-1j*kH*d_my) 
    elif np.isclose(d_my, 0, 1E-3):
        F =  I1 * ( 2*sin(kH*d_ny) / (kH*d_ny) + 
                   sum([ kH/sqrt(complex(kH**2 - (s*pi)**2))*
                        ( sin(kH*d_ny + s*pi)/(kH*d_ny + s*pi)+
                          sin(kH*d_ny - s*pi)/(kH*d_ny - s*pi)  )*
                          (sin(s*pi*Q_y/H) - sin(s*pi*P_y/H))/(s*pi)  for s in range(1,Np)]) )
    else:
        F =  I1 * ( sin(kH*d_ny) / (kH*d_ny) * (exp(-1j*kH*d_my*Q_y/H) - exp(-1j*kH*d_my*P_y/H)) / (-1j*kH*d_my) +
                  0.5*sum([ kH/sqrt(complex(kH**2 - (s*pi)**2))*
                        ( sin(kH*d_ny + s*pi)/(kH*d_ny + s*pi)+
                          sin(kH*d_ny - s*pi)/(kH*d_ny - s*pi)  )*
                          ( ( exp(-1j*(kH*d_my - s*pi)*Q_y/H) - exp(-1j*(kH*d_my - s*pi)*P_y/H)) / (-1j*(kH*d_my - s*pi)) +
                            ( exp(-1j*(kH*d_my + s*pi)*Q_y/H) - exp(-1j*(kH*d_my + s*pi)*P_y/H)) / (-1j*(kH*d_my + s*pi))
                                                                   )  for s in range(1,Np)]) )

    
    #second term
        
    I = -1j*kH*d_nN*exp(1j*kH*(d_nx-d_mx)*x/H)
    if np.isclose(d_ny, d_my, 1E-3):
        S = I*exp(1j*(d_ny-d_my)*kH)
    else:
        S = I * ( exp( 1j*kH*(d_ny-d_my)*Q_y/H) -exp( 1j*kH*(d_ny-d_my)*P_y/H)  ) / ( 1j*kH*(d_ny-d_my))

    centred = F+S

    # REGULARIZATION
    reg = 0
    return centred + d_2*reg



def AssembleMatrix(V : TrefftzSpace, Edges : tuple[Edge], 
                   a = 0.5, b = 0.5, d_1 = 0.5, d_2 = 0.5, 
                   Np=10, fullsides=False, sparse = False) -> spmatrix:
    '''Assembles de matrix for the bilinear form.
    a, b, d_1 and d_2 are the coefficients of the regularizing terms.
    Use fullsides = True if each boundary Sigma_R+ and Sigma_R- consists of 
    a single triangle side.'''
    if sparse:
        if fullsides:
            print(f'{a=}, {b=}, {d_1= }, {d_2=}')
            return AssembleMatrix_full_sides_sparse(V, Edges, a, b, d_1, d_2, Np)
        else:
            return AssembleMatrix_broken_sides_sparse(V, Edges, a, b, d_1, d_2, Np)
    else:
        return AssembleMatrix_full_sides(V, Edges, a, b, d_1, d_2, Np)

    return

def AssembleMatrix_broken_sides_sparse(V, Edges, a, b, d_1, d_2, Np):
    '''Assembles de matrix assuming Sigma_R+- consist of several triangles sides,
    and returns a Scipy sparse matrix.'''   
                    
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
                        for m in V.DOF_range[K_other]:
                            psi = Psi[m]
                            i_index.append(m)
                            j_index.append(n)
                            values.append(Sigma_term(phi, psi, E, d_2, Np=Np))
                        
    values = np.array(values)
    i_index = np.array(i_index)
    j_index = np.array(j_index)
    
    
    A = coo_matrix( (values, (i_index, j_index)), shape=(N_DOF,N_DOF))
    A = csr_matrix(A)

    return A





def AssembleMatrix_full_sides(V, Edges, a, b, d_1, d_2, Np=10):

    N_DOF = V.N_DOF
    A = np.zeros((N_DOF,N_DOF), dtype=np.complex128)
   
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
                        A[m,n] += Inner_term(phi, psi, E, a, b)

                for n in V.DOF_range[K_minus]:
                    phi = Phi[n]
                    for m in V.DOF_range[K_plus]:
                        psi = Psi[m]
                        A[m,n] += Inner_term(phi, psi, E, -a, -b)

                for n in V.DOF_range[K_plus]:
                    phi = Phi[n]
                    for m in V.DOF_range[K_minus]:
                        psi = Psi[m]
                        A[m,n] += -Inner_term(phi, psi, E, a, b)

                for n in V.DOF_range[K_minus]:
                    phi = Phi[n]
                    for m in V.DOF_range[K_minus]:
                        psi = Psi[m]
                        A[m,n] += -Inner_term(phi, psi, E, -a, -b)


            case EdgeType.GAMMA:
                K = E.Triangles[0]
                for m in V.DOF_range[K]:
                    psi = Psi[m]
                    for n in V.DOF_range[K]:
                        phi = Phi[n]
                        A[m,n] += Gamma_term(phi, psi, E, d_1)
                    

            case EdgeType.D_OMEGA:
                K = E.Triangles[0]
                for m in V.DOF_range[K]:
                    psi = Psi[m]
                    for n in V.DOF_range[K]:
                        phi = Phi[n]
                        A[m,n] += sound_soft_term(phi, psi, E, a)


            case EdgeType.SIGMA_L:
                K = E.Triangles[0]
                for n in V.DOF_range[K]:
                    phi = Phi[n]
                    for m in V.DOF_range[K]:
                        psi = Psi[m]
                        A[m,n] += Sigma_term(phi, psi, E, d_2, Np=Np)
            case EdgeType.SIGMA_R:
                K = E.Triangles[0]
                for n in V.DOF_range[K]:
                    phi = Phi[n]
                    for m in V.DOF_range[K]:
                        psi = Psi[m]
                        A[m,n] += Sigma_term(phi, psi, E, d_2, Np=Np)


    return A

def AssembleMatrix_full_sides_sparse(V, Edges, a, b, d_1, d_2, Np=10) -> spmatrix:
    '''Assembles de matrix assuming Sigma_R+- consist of a single triangle side,
    and returns a Scipy sparse matrix.'''

    N_DOF = V.N_DOF

    values = []
    i_index = []
    j_index = []
   
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
                    for m in V.DOF_range[K]:
                        psi = Psi[m]
                        i_index.append(m)
                        j_index.append(n)
                        values.append(Sigma_term(phi, psi, E, d_2, Np=Np))


    A = coo_matrix( (np.array(values), (np.array(i_index), np.array(j_index))), shape=(N_DOF,N_DOF))
    A = csr_matrix(A)

    return A



def exact_RHS(psi, E, k, H, d_2, t=0): 
    d = psi.d
    d_x = d[0]
    d_y = d[1]
    N = E.N

    x = E.P[0]
    kH = k*H

    beta = sqrt(complex(kH**2 - (t*pi)**2))

    I = 2*1j*k*H*exp(1j*(beta-kH*d_x)*x/H)
    if np.isclose(d_y,0,1E-3):
        if t == 0:
            F = 2*I*(dot(d,N) - d_2)
        else:
            F =  0.
    else:
        if t == 0:
            F = 2*I*(dot(d,N) - d_2)* sin(kH*d_y)/(kH*d_y)
        else:
            F = I*(dot(d,N) - d_2)*( sin(kH*d_y+t*pi)/(kH*d_y+t*pi) +  sin(kH*d_y-t*pi)/(kH*d_y-t*pi) )


    if np.isclose(d_y,0,1E-3):
        if t == 0:
            S = 2*I*(dot(d,N)*d_2)
        else:
            S =  0.
    else:
        if t == 0:
            S = 2*I*(dot(d,N)*d_2)*sin(kH*d_y)/(kH*d_y)
        else:
            S = I*(dot(d,N)*d_2)*kH/conj(beta)*( sin(kH*d_y+t*pi)/(kH*d_y+t*pi) +  sin(kH*d_y-t*pi)/(kH*d_y-t*pi) )

    return F + S

def exact_RHS_broken(psi, E, k, H, d_2, t=0, Np=15):
    d = psi.d
    # d_x = d[0]
    d_y = d[1]
    N = E.N
    M = E.M
    l = E.l

    # x = E.P[0]
    # kH = k*H

    beta = sqrt(complex(k**2 - (t*pi/H)**2))

    F = 1j*k*l*exp(1j*beta*M[0])*exp(-1j*k*dot(d,M))*(dot(d,N) - d_2)*(exp( 1j*pi*t/H*M[1])*sinc(t*l/(2*H) - k*l*d_y/(2*pi)) + 
                                                                       exp(-1j*pi*t/H*M[1])*sinc(t*l/(2*H) + k*l*d_y/(2*pi)))


    S = 1j*k*l*exp(1j*beta*M[0])*exp(-1j*k*dot(d,M))*dot(d,N)*d_2*l/H*( sinc(k*l*d[1]/(2*pi))*sinc(t*l/(2*H))*cos(pi*t/H*M[1]) + 
                                                                        1/2* sum( [ k/conj(sqrt(complex(k**2 - (p*pi/H)**2))) *
                                                                        (exp( 1j*p*pi*M[1]/H)*sinc(p*l/(2*H) - k*l*d[1]/(2*pi)) + 
                                                                         exp(-1j*p*pi*M[1]/H)*sinc(p*l/(2*H) + k*l*d[1]/(2*pi))) * 
                                                                        (sinc((t+p)*l/(2*H))*cos((t+p)*pi/H*M[1]) + 
                                                                         sinc((t-p)*l/(2*H))*cos((t-p)*pi/H*M[1])) 
                                                                        for p in range(1,Np)] ) )


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
                    if full_sides:
                        b[m] += exact_RHS(psi, E, k, H, d_2, t=t)
                    else:
                        b[m] += exact_RHS_broken(psi, E, k, H, d_2, t = t)
            case EdgeType.SIGMA_R:
                pass
    return b

