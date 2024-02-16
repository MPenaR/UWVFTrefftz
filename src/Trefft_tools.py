
import numpy as np
from numpy import dot, pi, exp, sqrt, sin, abs, conj
from numpy.linalg import norm
from collections import namedtuple
from labels import EdgeType

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



def Gamma_term(phi, psi, edge, k, d_1):

    d_m = psi.d
    d_n = phi.d
    
    P = edge.P
    Q = edge.Q
    N = edge.N
    T = edge.T

    l = norm(Q-P)

    I = (1 + d_1 * dot(d_n, N))*dot(d_m, N)

    if np.isclose( dot(d_m,T), dot(d_n,T), 1E-3) :
        return -1j*k*l* I * exp(1j*k*dot(d_n - d_m, P))
    else:
        return -I / dot(d_n - d_m, T) * ( exp(1j*k*dot(d_n - d_m, Q)) - exp(1j*k*dot(d_n - d_m, P)))
    



def Inner_term_PP(phi, psi, edge, k, a, b):

    d_m = psi.d
    d_n = phi.d
    
    P = edge.P
    Q = edge.Q
    N = edge.N
    T = edge.T

    l = norm(Q-P)

    I = dot( d_m, N) + dot( d_n, N) + 2*b*dot( d_m, N)*dot( d_n, N) + 2*a


    if np.isclose( dot(d_m,T), dot(d_n,T), 1E-3) :
        return -1/2*1j*k*l * I * exp(1j*k*dot(d_n - d_m, P))
    else:
        return -1/2*I/dot(d_n - d_m, T)*( exp(1j*k*dot(d_n - d_m, Q)) - exp(1j*k*dot(d_n - d_m, P)))


def Inner_term_PM(phi, psi, edge, k, a, b):

    d_m = psi.d
    d_n = phi.d
    
    P = edge.P
    Q = edge.Q
    N = edge.N
    T = edge.T

    l = norm(Q-P)

    I = dot( d_m, N) + dot( d_n, N) + 2*b*dot( d_m, N)*dot( d_n, N) + 2*a


    if np.isclose( dot(d_m,T), dot(d_n,T), 1E-3) :
        return 1/2*1j*k*l * I* exp(1j*k*dot(d_n - d_m, P))
    else:
        return 1/2*I/dot(d_n - d_m, T)*( exp(1j*k*dot(d_n - d_m, Q)) - exp(1j*k*dot(d_n - d_m, P)))


def Inner_term_MP(phi, psi, edge, k, a, b):

    d_m = psi.d
    d_n = phi.d
    
    P = edge.P
    Q = edge.Q
    N = edge.N
    T = edge.T

    l = norm(Q-P)

    I = dot( d_m, N) + dot( d_n, N) - 2*b*dot( d_m, N)*dot( d_n, N) - 2*a


    if np.isclose( dot(d_m,T), dot(d_n,T), 1E-3) :
        return -1/2*1j*k*l * I* exp(1j*k*dot(d_n - d_m, P))
    else:
        return -1/2*I/dot(d_n - d_m, T)*( exp(1j*k*dot(d_n - d_m, Q)) - exp(1j*k*dot(d_n - d_m, P)))


def Inner_term_MM(phi, psi, edge, k, a, b):

    d_m = psi.d
    d_n = phi.d
    
    P = edge.P
    Q = edge.Q
    N = edge.N
    T = edge.T

    l = norm(Q-P)

    I = dot( d_m, N) + dot( d_n, N) - 2*b*dot( d_m, N)*dot( d_n, N) - 2*a


    if np.isclose( dot(d_m,T), dot(d_n,T), 1E-3) :
        return 1/2*1j*k*l * I* exp(1j*k*dot(d_n - d_m, P))
    else:
        return 1/2*I/dot(d_n - d_m, T)*( exp(1j*k*dot(d_n - d_m, Q)) - exp(1j*k*dot(d_n - d_m, P)))






def Sigma_term(phi, psi, edge, k, H, d_2, Np = 15):

    d_n = phi.d
    d_m = psi.d

    d_mx = d_m[0]
    d_my = d_m[1]
    d_nx = d_n[0]
    d_ny = d_n[1]
    


    kH = k*H

    P = edge.P
    N = edge.N
    x  = P[0]/H

    d_nN = dot(d_n,N)
    d_mN = dot(d_m,N)
    
    #first term
    I1 = -2*1j*kH*exp(1j*(d_nx-d_mx)*kH*x)*d_mN*d_nN
   
    if np.isclose(d_ny, 0, 1E-3) and np.isclose(d_my, 0, 1E-3):
        F = I1
    elif np.isclose(d_ny, 0, 1E-3):
        F = I1 * sin(d_my*kH) / (d_my*kH)
    elif np.isclose(d_my, 0, 1E-3):
        F =  I1 * sin(d_ny*kH) / (d_ny*kH)
    else:
        F = I1 * sin(d_my*kH)/(d_my*kH) * sin(d_ny*kH)/(d_ny*kH) + \
            I1 * 0.5*sum([kH/sqrt(complex(kH**2 - (s*pi)**2)) * (sin(d_ny*kH+s*pi)/(d_ny*kH+s*pi) + sin(d_ny*kH-s*pi)/(d_ny*kH-s*pi)) 
                                                              * (sin(d_my*kH+s*pi)/(d_my*kH+s*pi) + sin(d_my*kH-s*pi)/(d_my*kH-s*pi))  
                                                              for s in range(1,Np)])
        
    # cross-terms
    I1 = 2*1j*kH*exp(1j*(d_nx-d_mx)*kH*x)*d_2*d_nN
    
    if np.isclose(d_ny, 0, 1E-3) and np.isclose(d_my, 0, 1E-3):
        C1 = I1
    elif np.isclose(d_ny, 0, 1E-3):
        C1 = I1 * sin(d_my*kH) / (d_my*kH)
    elif np.isclose(d_my, 0, 1E-3):
        C1 =  I1 * sin(d_ny*kH) / (d_ny*kH)
    else:
        C1 = I1 * sin(d_my*kH)/(d_my*kH) * sin(d_ny*kH)/(d_ny*kH) + \
            I1 * 0.5*sum([kH/sqrt(complex(kH**2 - (s*pi)**2)) * (sin(d_ny*kH+s*pi)/(d_ny*kH+s*pi) + sin(d_ny*kH-s*pi)/(d_ny*kH-s*pi)) 
                                                              * (sin(d_my*kH+s*pi)/(d_my*kH+s*pi) + sin(d_my*kH-s*pi)/(d_my*kH-s*pi))  
                                                              for s in range(1,Np)])


    I1 = 2*1j*kH*exp(1j*(d_nx-d_mx)*kH*x)*d_2*d_mN
    
    if np.isclose(d_ny, 0, 1E-3) and np.isclose(d_my, 0, 1E-3):
        C2 = I1
    elif np.isclose(d_ny, 0, 1E-3):
        C2 = I1 * sin(d_my*kH) / (d_my*kH)
    elif np.isclose(d_my, 0, 1E-3):
        C2 =  I1 * sin(d_ny*kH) / (d_ny*kH)
    else:
        C2 = I1 * sin(d_my*kH)/(d_my*kH) * sin(d_ny*kH)/(d_ny*kH) + \
            I1 * 0.5*sum([kH/conj(sqrt(complex(kH**2 - (s*pi)**2))) * (sin(d_ny*kH+s*pi)/(d_ny*kH+s*pi) + sin(d_ny*kH-s*pi)/(d_ny*kH-s*pi)) 
                                                              * (sin(d_my*kH+s*pi)/(d_my*kH+s*pi) + sin(d_my*kH-s*pi)/(d_my*kH-s*pi))  
                                                              for s in range(1,Np)])



    #second-like terms
        
    I = -2*1j*kH*(d_nN+d_2)*exp(1j*(d_nx-d_mx)*kH*x)
    if np.isclose(d_ny, d_my, 1E-3):
        S = I
    else:
        S = I * sin((d_ny-d_my)*kH) / ((d_ny-d_my)*kH)

    #N(...)N(...) terms
    I1 = -2*1j*kH*exp(1j*(d_nx-d_mx)*kH*x)*d_2*d_mN*d_nN

    if np.isclose(d_ny, 0, 1E-3) and np.isclose(d_my, 0, 1E-3):
        NN = I1
    elif np.isclose(d_ny, 0, 1E-3):
        NN = I1 * sin(d_my*kH) / (d_my*kH)
    elif np.isclose(d_my, 0, 1E-3):
        NN =  I1 * sin(d_ny*kH) / (d_ny*kH)
    else:
        NN = I1 * sin(d_my*kH)/(d_my*kH) * sin(d_ny*kH)/(d_ny*kH) + \
             I1 * 0.5*sum([kH**2/abs(sqrt(complex(kH**2 - (s*pi)**2)))**2 
                           * (sin(d_ny*kH + s*pi)/(d_ny*kH + s*pi) + sin(d_ny*kH - s*pi)/(d_ny*kH - s*pi)) 
                           * (sin(d_my*kH + s*pi)/(d_my*kH + s*pi) + sin(d_my*kH - s*pi)/(d_my*kH - s*pi))  for s in range(1,Np)])
 

    return F + S + C1 + C2+ NN 


# def Sigma_separated(phi, psi, edge, k, H, d_2, Np = 15):

#     d_n = phi.d
#     d_m = psi.d

#     d_mx = d_m[0]
#     d_my = d_m[1]
#     d_nx = d_n[0]
#     d_ny = d_n[1]
    


#     kH = k*H
    
#     P = edge.P 
#     N = edge.N
#     x  = P[0]/H

#     d_nN = dot(d_n,N)
#     d_mN = dot(d_m,N)

#     #CENTRED FLUXES
    
#     #first-like terms
#     I1 = -2*1j*kH*exp(1j*(d_nx-d_mx)*kH*x)*d_mN*d_nN

#     if np.isclose(d_ny, 0, 1E-3) and np.isclose(d_my, 0, 1E-3):
#         F = I1 
#     elif np.isclose(d_ny, 0, 1E-3):
#         F = I1 * sin(d_my*kH) / (d_my*kH) 
#     elif np.isclose(d_my, 0, 1E-3):
#         F =  I1 * sin(d_ny*kH) / (d_ny*kH)
#     else:

#         F = I1 * (sin(d_my*kH) / (d_my*kH) * sin(d_ny*kH) / (d_ny*kH) +
#         1/2*sum([kH/sqrt(complex(kH**2 - (s*pi)**2)) * (sin(d_ny*kH + s*pi)/(d_ny*kH + s*pi) + sin(d_ny*kH - s*pi)/(d_ny*kH - s*pi)) 
#                                                      * (sin(d_my*kH + s*pi)/(d_my*kH + s*pi) + sin(d_my*kH - s*pi)/(d_my*kH - s*pi))  
#                                               for s in range(1,Np)]))

#     #second-like terms
        
#     I = -2*1j*kH*d_nN*exp(1j*(d_nx-d_mx)*kH*x)
#     if np.isclose(d_ny, d_my, 1E-3):
#         S = I 
#     else:
#         S = I * sin((d_ny-d_my)*kH) / ((d_ny-d_my)*kH)

#     centred = F+S


#     #REGULARIZATON


#     if np.isclose(d_ny, 0, 1E-3) and np.isclose(d_my, 0, 1E-3):
#         first =  2*d_nN*d_mN
#         second = -2*d_nN
#         third  = -2*d_mN
#     elif np.isclose(d_ny, 0, 1E-3):
#         first  = 2*d_nN*d_mN * sin(kH*d_my)/(kH*d_my)
#         second = -2*d_nN     * sin(kH*d_my)/(kH*d_my)
#         third  = -2*d_mN     * sin(kH*d_my)/(kH*d_my)
#     elif np.isclose(d_my, 0, 1E-3):
#         first  = 2*d_nN*d_mN * sin(kH*d_ny)/(kH*d_ny)
#         second = -2*d_nN     * sin(kH*d_ny)/(kH*d_ny)
#         third  = -2*d_mN     * sin(kH*d_ny)/(kH*d_ny)
#     else:
#         first = 2*d_nN*d_mN * ( sin(kH*d_ny)/(kH*d_ny) * sin(kH*d_my)/(kH*d_my) + 
#                           0.5 * sum( [ kH**2 / sqrt(complex(kH**2 - (s*pi)**2))**2 *
#                         (sin(kH*d_ny+s*pi)/(kH*d_ny+s*pi) + sin(kH*d_ny-s*pi)/(kH*d_ny-s*pi)) *
#                         (sin(kH*d_my+s*pi)/(kH*d_my+s*pi) + sin(kH*d_my-s*pi)/(kH*d_my-s*pi))
#                                          for s in range(1,Np)]) )

        
#         second = -2*d_nN * ( sin(kH*d_ny)/(kH*d_ny) * sin(kH*d_my)/(kH*d_my) + 
#                           0.5 * sum( [ kH / sqrt(complex(kH**2 - (s*pi)**2)) *
#                         (sin(kH*d_ny+s*pi)/(kH*d_ny+s*pi) + sin(kH*d_ny-s*pi)/(kH*d_ny-s*pi)) *
#                         (sin(kH*d_my+s*pi)/(kH*d_my+s*pi) + sin(kH*d_my-s*pi)/(kH*d_my-s*pi))
#                                          for s in range(1,Np)]) )

#         third  = -2*d_mN * ( sin(kH*d_ny)/(kH*d_ny) * sin(kH*d_my)/(kH*d_my) + 
#                           0.5 * sum( [ kH / sqrt(complex(kH**2 - (s*pi)**2)) *
#                         (sin(kH*d_ny+s*pi)/(kH*d_ny+s*pi) + sin(kH*d_ny-s*pi)/(kH*d_ny-s*pi)) *
#                         (sin(kH*d_my+s*pi)/(kH*d_my+s*pi) + sin(kH*d_my-s*pi)/(kH*d_my-s*pi))
#                                          for s in range(1,Np)]) )

#     if d_my == d_ny:
#         forth = 1.
#     else:
#         forth = sin(kH*(d_ny - d_my))/(kH*(d_ny - d_my))

    
#     reg = -1j*kH*exp(1j*kH*(d_nx-d_mx)*x)*( first + second + third + forth)

#     return centred + d_2*reg


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







def AssembleMatrix(V, Edges, k, H, a, b, d_1, d_2, Np=10):

    Sigma_R_edges = [] 
    Sigma_L_edges = [] 
    
    for E in Edges:
        match E.Type:
            case EdgeType.SIGMA_L:
                Sigma_L_edges.append(E)
            case EdgeType.SIGMA_R:
                Sigma_R_edges.append(E)    


    N_DOF = V.N_DOF
    A = np.zeros((N_DOF,N_DOF), dtype=np.complex128)
   
    Phi = V.TrialFunctions
    Psi = V.TestFunctions # currently the same spaces 
    for (s,E) in enumerate(Edges):
        match E.Type:
            case EdgeType.INNER:
                K_plus, K_minus = E.Triangles
                for n in V.DOF_range[K_plus]:
                    phi = Phi[n]
                    for m in V.DOF_range[K_plus]:
                        psi = Psi[m]      
                        A[m,n] += Inner_term_PP(phi, psi, E, k, a, b)

                for n in V.DOF_range[K_minus]:
                    phi = Phi[n]
                    for m in V.DOF_range[K_plus]:
                        psi = Psi[m]
                        A[m,n] += Inner_term_MP(phi, psi, E, k, a, b)

                for n in V.DOF_range[K_plus]:
                    phi = Phi[n]
                    for m in V.DOF_range[K_minus]:
                        psi = Psi[m]
                        A[m,n] += Inner_term_PM(phi, psi, E, k, a, b)

                for n in V.DOF_range[K_minus]:
                    phi = Phi[n]
                    for m in V.DOF_range[K_minus]:
                        psi = Psi[m]
                        A[m,n] += Inner_term_MM(phi, psi, E, k, a, b)


            case EdgeType.GAMMA:
                K = E.Triangles[0]
                for m in V.DOF_range[K]:
                    psi = Psi[m]
                    for n in V.DOF_range[K]:
                        phi = Phi[n]
                        A[m,n] += Gamma_term(phi, psi, E, k, d_1)
                    
            case EdgeType.SIGMA_L:
                K = E.Triangles[0]
                for n in V.DOF_range[K]:
                    phi = Phi[n]
                    # for K_j with edges in sigma_L
                    for E_j in Sigma_L_edges:
                        K_j = E_j.Triangles[0]
                        for m in V.DOF_range[K_j]:
                            psi = Psi[m]
                            A[m,n] += Sigma_broken(phi, psi, E, k, H, d_2, Np=Np)
            case EdgeType.SIGMA_R:
                K = E.Triangles[0]
                for n in V.DOF_range[K]:
                    phi = Phi[n]
                    # for K_j with edges in sigma_L
                    for E_j in Sigma_R_edges:
                        K_j = E_j.Triangles[0]
                        for m in V.DOF_range[K_j]:
                            psi = Psi[m]
                            #A[m,n] += Sigma_term(phi, psi, E, k, H, d_2, Np=Np)
                            #A[m,n] += Sigma_separated(phi, psi, E, k, H, d_2, Np=Np)
                            A[m,n] += Sigma_broken(phi, psi, E, k, H, d_2, Np=Np)
    return A


def AssembleMatrix_full_sides(V, Edges, k, H, a, b, d_1, d_2, Np=10):

    N_DOF = V.N_DOF
    A = np.zeros((N_DOF,N_DOF), dtype=np.complex128)
   
    Phi = V.TrialFunctions
    Psi = V.TestFunctions # currently the same spaces 
    for (s,E) in enumerate(Edges):
        match E.Type:
            case EdgeType.INNER:
                K_plus, K_minus = E.Triangles
                for n in V.DOF_range[K_plus]:
                    phi = Phi[n]
                    for m in V.DOF_range[K_plus]:
                        psi = Psi[m]      
                        A[m,n] += Inner_term_PP(phi, psi, E, k, a, b)

                for n in V.DOF_range[K_minus]:
                    phi = Phi[n]
                    for m in V.DOF_range[K_plus]:
                        psi = Psi[m]
                        A[m,n] += Inner_term_MP(phi, psi, E, k, a, b)

                for n in V.DOF_range[K_plus]:
                    phi = Phi[n]
                    for m in V.DOF_range[K_minus]:
                        psi = Psi[m]
                        A[m,n] += Inner_term_PM(phi, psi, E, k, a, b)

                for n in V.DOF_range[K_minus]:
                    phi = Phi[n]
                    for m in V.DOF_range[K_minus]:
                        psi = Psi[m]
                        A[m,n] += Inner_term_MM(phi, psi, E, k, a, b)


            case EdgeType.GAMMA:
                K = E.Triangles[0]
                for m in V.DOF_range[K]:
                    psi = Psi[m]
                    for n in V.DOF_range[K]:
                        phi = Phi[n]
                        A[m,n] += Gamma_term(phi, psi, E, k, d_1)
                    
            case EdgeType.SIGMA_L:
                K = E.Triangles[0]
                for n in V.DOF_range[K]:
                    phi = Phi[n]
                    for m in V.DOF_range[K]:
                        psi = Psi[m]
                        A[m,n] += Sigma_term(phi, psi, E, k, H, d_2, Np=Np)
                        #A[m,n] += Sigma_separated(phi, psi, E, k, H, d_2, Np=Np)
            case EdgeType.SIGMA_R:
                K = E.Triangles[0]
                for n in V.DOF_range[K]:
                    phi = Phi[n]
                    for m in V.DOF_range[K]:
                        psi = Psi[m]
                        A[m,n] += Sigma_term(phi, psi, E, k, H, d_2, Np=Np)
                        #A[m,n] += Sigma_separated(phi, psi, E, k, H, d_2, Np=Np)
    return A






def exact_RHS(psi, E, k, H, d_2, t=0): 
    d = psi.d
    d_x = d[0]
    d_y = d[1]
    N = E.N

    x = E.P[0]/H
    kH = k*H

    if np.isclose(d_y,0,1E-3):
        if t == 0:
            return -4*1j*kH*exp(1j*k*H*(1-d_x)*x)*(d_x - d_2 - d_x*d_2)
        else:
            return 0.
    else:
        if t == 0:
            return -4*1j*kH*exp(1j*k*H*(1-d_x)*x)*(d_x - d_2 - d_x*d_2 )* sin(kH*d_y)/(kH*d_y)
        else:
            return -2*1j*kH*exp(1j*(sqrt(complex(kH**2 - (t*pi)**2))-kH*d_x)*x)*\
                (d_x - d_2 - d_x*d_2 * kH / sqrt(complex(kH**2 - (t*pi)**2)) ) *\
                    ( sin(kH*d_y+t*pi)/(kH*d_y+t*pi) +  sin(kH*d_y-t*pi)/(kH*d_y-t*pi) )




def exact_separated(psi, E, k, H, d_2, t=0): 
    d = psi.d
    d_x = d[0]
    d_y = d[1]
    N = E.N
    d_N = dot(d,N)

    x = E.P[0]/H
    kH = k*H

    beta = sqrt( complex( kH**2 - (t*pi)**2))


    #centred part 


    if np.isclose(d_y,0,1E-3):
        if t == 0:
            centred =  2.
        else:
            centred = 0.
    else:
        if t == 0:
            centred = 2 * sin(kH*d_y)/(kH*d_y)
        else:
            centred = (sin( kH*d_y + t*pi) / ( kH*d_y + t*pi) +
                       sin( kH*d_y - t*pi) / ( kH*d_y - t*pi)  )
            
    centred = 2j *kH* d_N * exp(1j*(beta-kH*d_x)*x) * centred 
        
    # d_2 part
            
    
    if np.isclose(d_y,0,1E-3):
        if t == 0:
            reg1 =  2*d_N
            reg2 = -2
        else:
            reg1 =  0.
            reg2 =  0.
    
    else:
        if t == 0:
            reg1 = d_N * sin(kH*d_y)/(kH*d_y)
            reg2 = - 2 * sin(kH*d_y)/(kH*d_y)
        else:
            reg1 = d_N * kH / beta * ( sin( kH*d_y + t*pi) / ( kH*d_y + t*pi) +
                                       sin( kH*d_y - t*pi) / ( kH*d_y - t*pi) )
        
            reg2 = - ( sin( kH*d_y + t*pi) / ( kH*d_y + t*pi) +
                       sin( kH*d_y - t*pi) / ( kH*d_y - t*pi) )

    
    reg = 2j *kH*exp(1j*(beta-kH*d_x)*x)*( reg1 + reg2)
    
    return centred + d_2*reg


def exact_RHS_broken(psi, E, k, H, d_2, t=0):
    d = psi.d
    P_y = E.P[1]
    Q_y = E.Q[1]

    d_x = d[0]
    d_y = d[1]
    
    R = 10 #!! REPLACE DO NOT HARDCODE
    
    #N = E.N
    # d_N = dot(d,N)

    kH = k*H

    beta = sqrt( complex( kH**2 - (t*pi)**2))


    #centred part

    if np.isclose(d_y,0,1E-3):
        if t == 0:
            I =  (Q_y - P_y)/H
        else:
            I =  (sin(t*pi*Q_y/H) - sin(t*pi*P_y/H))/(t*pi)
    else:
        if t == 0:
            I = - 1/(1j*kH*d_y)*( exp(-1j*kH*d_y*Q_y/H) - exp(-1j*kH*d_y*P_y/H))
        else:
            I = -((exp(-1j*(kH*d_y - t*pi)*Q_y/H) - exp(-1j*(kH*d_y - t*pi)*P_y/H) ) / (2j*(kH*d_y - t*pi)) +
                  (exp(-1j*(kH*d_y + t*pi)*Q_y/H) - exp(-1j*(kH*d_y + t*pi)*P_y/H) ) / (2j*(kH*d_y + t*pi))  )
            
    centred = -2j*kH*d_x*exp(1j*(kH*d_x - beta)*R/H)*I

    reg = 0.

    return centred + d_2*reg





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
                    # b[m] += exact_separated(psi, E, k, H, d_2, t = t)
                    else:
                        b[m] += exact_RHS_broken(psi, E, k, H, d_2, t = t)
            case EdgeType.SIGMA_R:
                pass
    return b

