"""
module for the numpy based mesh class.
As far as I'm concerned a mesh should consist of an array of vertices
and an array of edges.

"""
import numpy as np
from geometry import midpoint, normal, tangent, length 


edge_dt = np.dtype([("N",np.float64,(2,)),
                    ("T",np.float64,(2,)),
                    ("l",np.float64),
                    ("M",np.float64,(2,))])



class Mesh():

# Edges = np.array( [ ( E.N, E.T, E.l, E.M) for E in [ Edge(Omega, e)  for e in Omega.edges ] ], dtype=edge_dt)
