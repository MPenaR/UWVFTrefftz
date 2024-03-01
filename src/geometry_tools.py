
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
    
    elif edge.nr in [ e.edges[0].nr for e in Omega.Boundaries("D_Omega").Elements()]:
        return EdgeType.D_OMEGA
    else:
        return EdgeType.INNER


c = (0,0) #NEED TO FIX THIS

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
        
        self.N = self.getNormal()
        self.T = self.getTangent() 
        self.midpoint = (P+Q)/2 #not needed right now
        self.Triangles = self.setTriangles(Omega,edge)


#        probably they should be properties with getters and setters, fix later    

    def getNormal(self):
        px, py = self.P 
        qx, qy = self.Q 
        tx, ty = self.Q - self.P

        match self.Type: #maybe use gamma_up and gamma_down
            
            case EdgeType.GAMMA:
                return np.array([0., py / np.abs(py)])
            
            case EdgeType.SIGMA_L:
                return np.array([-1., 0.])

            case EdgeType.SIGMA_R:
                return np.array([1., 0.])

            case EdgeType.INNER:
                return np.array([ -ty, tx] ) / norm([tx,ty])
            
            case EdgeType.D_OMEGA:
                N = np.array([ -ty, tx] ) / norm([tx,ty])
                if dot((self.Q+self.P)/2 - np.array(c),N) > 0:  # need to fix this
                    N = -N
                return N


    def getTangent(self):
        T = (self.Q - self.P) / norm(self.Q - self.P)
        return T

    def setTriangles(self, Omega, edge):
        Triangles = [ K.nr for K in edge.faces ]
        if len(Triangles) == 1:
            return Triangles
        else:  # put first K^+
            B_1 = 1/3*(sum([ np.array(Omega.vertices[v.nr].point) for v in Omega.faces[Triangles[0]].vertices ]))          
            B_2 = 1/3*(sum([ np.array(Omega.vertices[v.nr].point) for v in Omega.faces[Triangles[1]].vertices ]))          
            if dot(self.N, B_2 - B_1) > 0:
                return Triangles
            else:
                return [Triangles[1], Triangles[0]]

