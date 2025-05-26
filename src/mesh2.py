"""module for defining the mesh class"""

import numpy as np
import numpy.typing as npt
from matplotlib.tri import Triangulation

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

    @staticmethod
    def from_numpy(cls, points: float_array,
                   edges: int_array,
                   faces: int_array):
        """it assumes the points/edges/faces IDs are their row index.
        edges is a  n_edges x 2 array, refering to the index of its ends
        faces is a n_faces x 3 array, refereing to the index of its edges"""

        vertices = np.arange(len(points), dtype=np.int64)
        edges = edges
        faces = faces
        return cls(points, vertices, edges, faces)

    @staticmethod
    def from_netgen(cls, points, edges, faces):
        pass

    @staticmethod
    def from_Triangulation(cls, tri: Triangulation):
        """
        Expects a Matplotlib triangulation
        """

        points = np.column_stack([tri.x, tri.y])
        vertices =  np.arange(len(points), dtype=np.int64)
        edges = tri.edges
        triangles


        return cls(points, vertices, edges, faces )

    def to_matplotlib(self):
        x = self.points[self.vertices["ID"], 0]
        y = self.points[self.vertices["ID"], 1]
        Triangles = self.faces["vertices"]
        return Triangulation(x, y, Triangles)



def generate_test_mesh() -> SurfaceMesh:
    points = np.array([[0, 0],
                   [1, 0],
                   [1, 1],
                   [0, 1]], dtype=np.float64)

    edges = np.array([[0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 0],
                    [0, 2]], dtype=np.int32) 

    faces = np.array([[0, 1, 2],
                    [0, 2, 3]], dtype=np.int32)
