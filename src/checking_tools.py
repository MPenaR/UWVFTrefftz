from netgen.geom2d import SplineGeometry
from ngsolve import Mesh
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from numpy import dot, exp, sqrt, pi, sin

from labels import EdgeType



def checkLabels(Edges, ax = None, R=10, H=1):
    """Checks the labelling"""

    if ax is None:
        _, ax = plt.subplots( figsize=(15,3))


    lw = 2

    for E in Edges:
        px, py = E.P
        qx, qy = E.Q
        match E.Type:
            case EdgeType.INNER:
                ax.plot([px, qx], [py, qy], 'k')

            case EdgeType.GAMMA:
                ax.plot([px, qx], [py, qy], 'g', linewidth=lw)

            case EdgeType.SIGMA_L:
                ax.plot([px, qx], [py, qy], '--r', linewidth=lw)

            case EdgeType.SIGMA_R:
                ax.plot([px, qx], [py, qy], '--r', linewidth=lw)
 
            case EdgeType.D_OMEGA:
                ax.plot([px, qx], [py, qy], 'b', linewidth=lw)


    d = 0.2
    ax.axis('square')
    ax.set_xlim([-R-d,R+d])
    ax.set_ylim([-H-d,H+d])


def checkNormals(Edges, R=10., H=1.):
    Normals = np.array( [ E.N for E in Edges])
    Tangents = np.array( [ E.T for E in Edges])
    MidPoints = np.array( [ E.midpoint for E in Edges])
    checkLabels(Edges)
    d = 0.2
    plt.quiver( MidPoints[:,0], MidPoints[:,1], Normals[:,0], Normals[:,1], scale=40)
    plt.quiver( MidPoints[:,0], MidPoints[:,1], Tangents[:,0], Tangents[:,1], scale=40, color='b')
    plt.xlim([-R-d,R+d])
    plt.ylim([-H-d,H+d])

# def checkPlusMinus(Edges,Baricenters):
#     Normals = np.array( [ E.N for E in Edges])
#     MidPoints = np.array( [ E.midpoint for E in Edges])
#     B_plus = np.array([Baricenters[E.Triangles[0]] for E in Edges])
#     NotNormals = -np.array([ (p-q)/norm(p-q) for (p,q) in zip(B_plus,MidPoints)])
#     checkLabels(Edges)
#     plt.quiver( MidPoints[:,0], MidPoints[:,1], Normals[:,0], Normals[:,1], scale=40, alpha = 0.2)
#     e = 0.2
#     for (m,n) in zip(MidPoints, NotNormals):
#         plt.text( m[0]+e*n[0], m[1]+ e*n[1], '-')
#         plt.text( m[0]-e*n[0], m[1]- e*n[1], '+')


