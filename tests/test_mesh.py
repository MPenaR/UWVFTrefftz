import numpy as np
from DGTrefftz.mesh import SurfaceMesh
from matplotlib.tri import Triangulation
from netgen.geom2d import SplineGeometry
from ngsolve import Mesh

def test_from_Triangulation():
    points = np.array([[0, 0],
                       [1, 0],
                       [1, 1],
                       [0, 1]], dtype=np.float64)

    triangles = np.array([[0, 1, 2],
                          [0, 2, 3]], dtype=np.int64)

    tri = Triangulation(x=points[:, 0], y=points[:, 1], triangles=triangles)
    S = SurfaceMesh.from_Triangulation(tri)
    faces = S.faces
    faces_expected = np.array([[0, 1, 2],
                               [1, 3, 4]], dtype=np.int64)
    x = np.array([0.3,0.6])
    y = np.array([0.6,0.3])
    print(S.in_triangle(x,y))
    assert np.all(faces == faces_expected)

def test_from_netgen():
    geo = SplineGeometry()
    geo.AddRectangle((0., 0.), (1., 1.))
    M = Mesh(geo.GenerateMesh())
    S = SurfaceMesh.from_netgen(M)
    faces = S.faces
    faces_expected = np.array([[1, 3, 0],
                               [2, 4, 1]], dtype=np.int64)
    x = np.array([0.3,0.6])
    y = np.array([0.6,0.3])
    print(S.in_triangle(x,y))
    assert np.all(faces == faces_expected)




# from block_fluxes import SoundHard_block

# NP = 3
# d_1 = 0.5
# k = 0.8




#element_dtype = [("face", "i4"), ("edges", "i4", (3)), ("points", "i4", (3))]
#
#elements = np.array( [(0, [0, 1, 4], [0, 1, 2]),
#                      (0, [4, 2, 3], [0, 2, 3])], dtype=element_dtype)

# A_wall = np.zeros([2*NP, 2*NP], dtype=np.complex128)


# theta_0 = 0
# theta = np.linspace(0,2*np.pi,NP) + theta_0
# d = np.column_stack([np.cos(theta), np.sin(theta)])

# d_d = np.zeros([NP,NP,2])
# d_d[:,:,0] = - np.subtract.outer(np.cos(theta), np.cos(theta))
# d_d[:,:,1] = - np.subtract.outer(np.sin(theta), np.sin(theta))
    
#     l = edge.l
#     N = edge.N
#     T = edge.T
#     M = edge.M


# A_wall[:NP,:NP] = SoundHard_block( k=k, edge=e0, d=d, d_d=d_d, d_1=d_1)
# A_wall[NP:,NP:] = SoundHard_block( k=k, edge=e2, d=d, d_d=d_d, d_1=d_1)
