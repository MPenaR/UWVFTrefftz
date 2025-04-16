"""
module for the numpy based mesh class.
As far as I'm concerned a mesh should consist of an array of vertices
and an array of edges. Triangles should also be numbered

"""
import numpy as np
from geometry import midpoint, normal, tangent, length 
from ngsolve import Mesh
from numpy_types import real_array, int_array
from numpy.typing import NDArray

edge_dt = np.dtype([("N", np.float64, (2,)),
                    ("T", np.float64, (2,)),
                    ("l", np.float64),
                    ("M", np.float64, (2,))])

point_dt = np.dtype([("xy", np.float64, (2,))])


class Mesh():
    def from__ngsolve(self, mesh : Mesh):
        self.vertices = np.array([v.point for v in mesh.vertices] , dtype=point_dt)
#         self.edges = np.array( [ ( E.N, E.T, E.l, E.M) for E in [ Edge(Omega, e)  for e in Omega.edges ] ], dtype=edge_dt)
        # self.edges = edges_from_
    
    def triangle( x : real_array, y : real_array ) -> int_array:

        


# def edges_from_ngsolve( mesh : Mesh ) -> real_array :
#     extracter = lambda E
#     

