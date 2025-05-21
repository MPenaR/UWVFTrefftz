"""module for defining the mesh class"""

import numpy as np 
import numpy.typing as npt

vertex_dt = [("ID", "i64")] #the ID is important so I can define a subset of vertices without changing their IDs
edge_dt = [ ("ID", "i64"), ("vertices", "i64", (2)) ]
face_dt = [ ("ID", "i64"), ("edges", "i64", (3))]


class SurfaceMesh:
    def __init__(self, points, vertices, edges, faces):
        self.points = points
        self.vertices = vertices
        self.edges = edges
        self.faces = faces 
    
    def __init__(self, points, edges, faces):
        self.points = points
        self.vertices = np.array()
        self.edges = edges
        self.faces = faces 


    def to_matplotlib(self):
        x = self.points[self.vertices["ID"],0]
        y = self.points[self.vertices["ID"],1]
        Triangles = np.array([])
        return (x, y, Triangles)



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
