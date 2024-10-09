import numpy as np
from numpy import dot
from numpy.linalg import norm
from labels import EdgeType

def computeEdgeType(Omega, edge):
        
    if edge.nr in [ e.edges[0].nr for e in Omega.Boundaries("Gamma").Elements()]:
        return EdgeType.GAMMA

    elif edge.nr in [ e.edges[0].nr for e in Omega.Boundaries("Sigma_L").Elements()]:
        return EdgeType.SIGMA_L

    elif edge.nr in [ e.edges[0].nr for e in Omega.Boundaries("Sigma_R").Elements()]:
        return EdgeType.SIGMA_R
    
    elif edge.nr in [ e.edges[0].nr for e in Omega.Boundaries("Cover").Elements()]:
        return EdgeType.COVER
    
    elif edge.nr in [ e.edges[0].nr for e in Omega.Boundaries("D_Omega").Elements()]:
        return EdgeType.D_OMEGA
    else:
        return EdgeType.INNER


class Edge:
    '''Holds all the information from an edge that is needed:
    - P : inital vertex
    - Q : end point
    - N : normal
    - T : tangent
    - Type: EdgeType: {INNER, GAMMA, SIGMA_L, SIGMA_R}
    - Triangless: tuple of one or two element IDs.'''

    def __init__(self, Omega, edge ):
        P = np.array( Omega.vertices[edge.vertices[0].nr].point)
        Q = np.array( Omega.vertices[edge.vertices[1].nr].point)        
        self.P = P
        self.Q = Q
        self.Type = computeEdgeType(Omega,edge)

        self.l = norm(Q-P)
        self.T = (Q - P) / self.l
        self.M = (P+Q)/2

        self.N = self.getNormal(Omega, edge)
        self.Triangles = self.setTriangles(Omega,edge)


    #probably they should be properties with getters and setters, fix later    
    def getNormal(self, Omega, edge):
        _, py = self.P 
        tx, ty = self.T

        match self.Type: #maybe use gamma_up and gamma_down
            
            case EdgeType.GAMMA: 
                if py == 0:
                    return np.array([0,-1.])
                else: 
                    return np.array([0,1.])
            
            case EdgeType.SIGMA_L | EdgeType.COVER:
                return np.array([-1., 0.])

            case EdgeType.SIGMA_R:
                return np.array([1., 0.])
            
            case EdgeType.D_OMEGA:
                V = np.array(Omega.vertices[({v.nr for v in Omega.faces[edge.faces[0].nr].vertices} - {v.nr for v in edge.vertices}).pop()].point)
                N = np.array([ -ty, tx] ) 
                if dot(V - self.M,N) > 0:  # NEED TO FIX THIS
                    N = -N
                return N

            case _:
                return np.array([ -ty, tx] )

    # def getTangent(self):
    #     T = (self.Q - self.P) / self.l
    #     return T

    def setTriangles(self, Omega, edge):
        Triangles = [ K.nr for K in edge.faces ]
        if len(Triangles) == 1:
            return Triangles
        else:  # put first K^+
            N = len(Omega.faces[Triangles[0]].vertices)
            B_1 = 1/N*(sum( np.array(Omega.vertices[v.nr].point) for v in Omega.faces[Triangles[0]].vertices ))          
            B_2 = 1/N*(sum( np.array(Omega.vertices[v.nr].point) for v in Omega.faces[Triangles[1]].vertices ))          
            if dot(self.N, B_2 - B_1) > 0:
                return Triangles
            return [Triangles[1], Triangles[0]]

class Triangle:
    "holds al the information for a Triangle, namely r_A, r_B and r_C"
    def __init__(self, Omega, f):
        self.A = np.array(Omega.vertices[f.vertices[0].nr].point)
        self.B = np.array(Omega.vertices[f.vertices[1].nr].point)
        self.C = np.array(Omega.vertices[f.vertices[2].nr].point)

        self.M = 1/3*( self.A + self.B + self.C)
        self.index = f.nr
