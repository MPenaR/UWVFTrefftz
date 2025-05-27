"""module for defining the mesh class"""

import numpy as np
import numpy.typing as npt
from matplotlib.tri import Triangulation
from itertools import combinations
from ngsolve import Mesh

float_array = npt.NDArray[np.float64]
complex_array = npt.NDArray[np.complex128]
int_array = npt.NDArray[np.int64]


# the ID is important so I can define a subset of vertices
# without changing their IDs
vertex_dt = [("ID", np.int64)]

DIM = 2

# edge_dt = [("ID", np.int64), 
#            ("P", np.float64, (DIM)), 
#            ("Q", np.float64, (DIM)), 
#            ("M", np.float64, (DIM)),
#            ("l", np.float64),
#            ("T", np.float64, (DIM)),
#            ("N", np.float64, (DIM))]]
edge_dt = [("ID", np.int64), 
           ("M", np.float64, (DIM)),
           ("l", np.float64),
           ("T", np.float64, (DIM)),
           ("N", np.float64, (DIM))]
# face_dt = [("ID", np.int64, ("edges", "i64", (3)), ("vertices", "i64", (3))]


class SurfaceMesh:
    def __init__(self, points, vertices, edges, faces):
        self.points = points
        self.vertices = vertices
        self.edges = edges
        self.faces = faces

    @classmethod
    def from_numpy(cls, points: float_array,
                   edges: int_array,
                   faces: int_array):
        """it assumes the points/edges/faces IDs are their row index.
        edges is a  n_edges x 2 array, refering to the index of its ends
        faces is a n_faces x 3 array, refering to the index of its edges"""

        vertices = np.arange(len(points), dtype=np.int64)
        edges = edges
        faces = faces
        return cls(points, vertices, edges, faces)

    @classmethod
    def from_netgen(cls, mesh: Mesh):
        """
        Expects a ngsolve mesh
        """
        points = np.array([v.point for v in mesh.vertices])
        vertices = np.arange(len(points), dtype=np.int64)
        edges = np.array([[e.vertices[0].nr, e.vertices[1].nr] for e in mesh.edges])
        faces = np.array([[f.edges[0].nr, f.edges[1].nr, f.edges[2].nr] for f in mesh.faces])
        return cls(points, vertices, edges, faces)

        

    @classmethod
    def from_Triangulation(cls, tri: Triangulation):
        """
        Expects a Matplotlib triangulation
        """

        points = np.column_stack([tri.x, tri.y])
        vertices = np.arange(len(points), dtype=np.int64)
        edges = tri.edges

        edges_dict = generate_edges_dict(edges)

        triangles = tri.triangles

        faces = np.zeros([len(triangles), 3], dtype=np.int64)
        for n, T in enumerate(triangles):
            faces[n] = [edges_dict[frozenset(c)] for c in combinations(T, 2)]

        return cls(points, vertices, edges, faces)

    def to_matplotlib(self) -> Triangulation:
        x = self.points[:, 0]
        y = self.points[:, 1]
        triangles = np.zeros((len(self.faces), 3), dtype=np.int64)
        edges = self.edges
        for n, f in enumerate(self.faces):
            triangles[n] = list(set(edges[f[0]]) | set(edges[f[1]]) | set(edges[f[2]]))
        return Triangulation(x, y, triangles)

    @property
    def edges_dict(self) -> dict[frozenset, int]:
        return generate_edges_dict(self.edges)

    def generate_edges_arrays(self):
        edges_array = np.zeros((len(self.edges)), dtype=edge_dt)
        for i, e in enumerate(self.edges):
            P = self.points[e[0]]
            Q = self.points[e[1]]
            M = 1/2*(P+Q)
            l = np.linalg.norm(P-Q)
            T = (P-Q)/l
            N = np.array([-T[1], T[0]])
            edges_array[i] = (i,M,l,T,N)
        return edges_array


# def generate_test_mesh() -> SurfaceMesh:


def generate_edges_dict(edges: int_array) -> dict[frozenset, int]:
    return {frozenset(e): i for i, e in enumerate(edges)}
