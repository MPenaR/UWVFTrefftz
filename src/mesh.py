"""module for defining the mesh class"""

import numpy as np
import numpy.typing as npt
from matplotlib.tri import Triangulation
from itertools import combinations

float_array = npt.NDArray[np.float64]
complex_array = npt.NDArray[np.complex128]
int_array = npt.NDArray[np.int64]


# the ID is important so I can define a subset of vertices
# without changing their IDs
vertex_dt = [("ID", "i64")]

edge_dt = [("ID", "i64"), ("vertices", "i64", (2))]
face_dt = [("ID", "i64"), ("edges", "i64", (3)), ("vertices", "i64", (3))]


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
    def from_netgen(cls, points, edges, faces):
        pass

  
    @classmethod
    def from_Triangulation(cls, tri: Triangulation):
        """
        Expects a Matplotlib triangulation
        """

        points = np.column_stack([tri.x, tri.y])
        vertices =  np.arange(len(points), dtype=np.int64)
        edges = tri.edges
        
        edges_dict = generate_edges_dict(edges)

        triangles = tri.triangles

        faces = np.zeros([len(triangles), 3], dtype=np.int64)
        for n, T in enumerate(triangles):
            faces[n] = [edges_dict[frozenset(c)] for c in combinations(T, 2)]



        return cls(points, vertices, edges, faces )

    def to_matplotlib(self) -> Triangulation:
        x = self.points[:, 0]
        y = self.points[:, 1]
        triangles = np.zeros((len(self.faces), 3), dtype=np.int64)
        edges = self.edges
        for n, f in enumerate(self.faces):
            triangles[n] = list(set(edges[f[0]]) | set(edges[f[1]]) | set(edges[f[2]]))
        return Triangulation(x, y, triangles)

    @property
    def edges_dict(self) -> dict[frozenset,int]:
        return generate_edges_dict(self.edges)

#def generate_test_mesh() -> SurfaceMesh:


def generate_edges_dict(edges: int_array) -> dict[frozenset, int]:
    return {frozenset(e): i for i, e in enumerate(edges)}

if __name__ == "__main__":

    points = np.array([[0, 0],
                       [1, 0],
                       [1, 1],
                       [0, 1]], dtype=np.float64)

    edges = np.array([[0, 1],
                      [1, 2],
                      [2, 3],
                      [3, 0],
                      [0, 2]], dtype=np.int32) 

    triangles = np.array([[0, 1, 2],
                          [0, 2, 3]], dtype=np.int32)
    
    tri = Triangulation(x=points[:,0], y=points[:,1], triangles=triangles)
    S = SurfaceMesh.from_Triangulation(tri)
    print(S.edges)
    print(S.faces)

    tri_2 = S.to_matplotlib()
    print(f'{tri_2.triangles=}')
    print(f'{tri_2.x=}, {tri_2.y=}')
