# module for the different domains

from netgen.geom2d import SplineGeometry
from ngsolve import Mesh
from enum import Enum, auto
from geometry_tools import Edge, EdgeType, Triangle
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.collections import LineCollection

class ScattererType(Enum):
    '''Enumeration of the different scatterer types.'''
    PENETRABLE = auto()
    SOUND_SOFT = auto()
    SOUND_HARD = auto()
    ABSORBING = auto()

class ScattererShape(Enum):
    '''Enumeration of the different scatterer shapes.'''
    RECTANGLE = auto()
    CIRCLE = auto()



class Waveguide:
    def __init__(self, R = 10., H = 1., half_infinite = False):
        self.R = R 
        self.H = H
        self.geo = SplineGeometry()
        self.half_infinite = half_infinite
        if half_infinite:
            self.geo.AddRectangle(p1=(0,0), p2=( R, H), bcs=["Gamma","Sigma_R","Gamma","Cover"], leftdomain=1, rightdomain=0)
        else:
            self.geo.AddRectangle(p1=(-R,0), p2=( R, H), bcs=["Gamma","Sigma_R","Gamma","Sigma_L"], leftdomain=1, rightdomain=0)

        self.geo.SetMaterial (1, "Omega_e")
        self.scatterer_type = None
        self.scatterer_markers = [] 
        self.scatterer_patchs = []
        self.Edges = None
        self.Omega = None
        self.meshed = False
        self.ScattererTriangles = []

    def add_scatterer( self, scatterer_shape : ScattererShape, scatterer_type : ScattererType, params : list) :
        self.scatterer_type = scatterer_type
        kwargs = {"edgecolor" : "k", "facecolor" : "grey", "linewidth" : 4}
        match scatterer_shape:
            case ScattererShape.CIRCLE:
                c, r = params

                self.scatterer_markers.append(lambda  x, y, c=c, r=r: (x-c[0])**2 + (y-c[1])**2 <= r**2)
                self.scatterer_patchs.append( lambda c=c, r=r, kwargs=kwargs : Circle(xy=c, radius=r, **kwargs))

                match scatterer_type:
                    case ScattererType.PENETRABLE | ScattererType.ABSORBING :
                        self.geo.AddCircle(c=c, r=r, leftdomain=2, rightdomain=1)
                        self.geo.SetMaterial (2, "Omega_i")
                    case _:
                        self.geo.AddCircle(c=c, r=r, bc="D_Omega", leftdomain=0, rightdomain=1)
            case ScattererShape.RECTANGLE:
                c, width, height = params

                self.scatterer_markers.append(lambda x, y, c=c, width=width, height=height: np.logical_and( np.abs(x - c[0])<= width/2, np.abs(y - c[1])<= height/2 ))
                self.scatterer_patchs.append( lambda c=c, width=width, height=height, kwargs=kwargs :Rectangle(xy=(c[0] - width/2, c[1]-height/2), height=height, width=width, **kwargs))
                
                match scatterer_type:
                    case ScattererType.PENETRABLE | ScattererType.ABSORBING:
                        self.geo.AddRectangle(p1=(c[0]-width/2,c[1]-height/2), p2=(c[0]+width/2,c[1]+height/2), leftdomain=2, rightdomain=1)
                        self.geo.SetMaterial (2, "Omega_i")
                    case _:
                        self.geo.AddRectangle(p1=(c[0]-width/2,c[1]-height/2), p2=(c[0]+width/2,c[1]+height/2), bc="D_Omega", leftdomain=0, rightdomain=1)
                    

    def add_fine_mesh_region(self, factor = 0.9, h_min = 0.1):
            factor = 0.9
            p5 = self.geo.AppendPoint(0,(1 - factor)*self.H/2)
            p6 = self.geo.AppendPoint(0,(1 + factor)*self.H/2)
            self.geo.Append(["line", p5,p6], leftdomain=1, rightdomain=1, maxh=h_min)



    def in_scatterer(self, x, y):
        mask = np.full_like(x,fill_value=False, dtype=np.bool_) # CHECK FOR WHAT DXR SAID, THERE IS NO LIST COPY
        for marker in self.scatterer_markers:
            mask = np.logical_or(mask, marker(x,y))
        return mask


    def generate_mesh(self, h_max = 1.):
        self.Omega = Mesh(self.geo.GenerateMesh(maxh= h_max))
        self.Edges = [ Edge(self.Omega, e)  for e in self.Omega.edges ]
        self.meshed = True
        if self.scatterer_type == ScattererType.ABSORBING:
            self.ScattererTriangles = [ Triangle(self.Omega, self.Omega.faces[e.faces[0].nr]) for e in self.Omega.Elements() if e.mat == "Omega_i"]
        return 
    
    def plot_scatterer_triangles(self):
        fig, ax = plt.subplots()
        ax.add_patch(Rectangle(xy=(-self.R, 0),width=2*self.R, height=self.H, facecolor='none', edgecolor='k'))
        for T in self.ScattererTriangles:
            ax.add_patch(Polygon([T.A, T.B, T.C], edgecolor='k', facecolor='b'))
        ax.axis('square')




    def plot_field(self, X, Y, Z, show_edges = False, ax = None):
        if ax is None: 
            _, ax = plt.subplots( figsize=(15,3))
        mask = self.in_scatterer(X.ravel(),Y.ravel())
        match self.scatterer_type:
            case ScattererType.SOUND_HARD | ScattererType.SOUND_SOFT:
                Z = np.where(mask,np.nan,Z.ravel()).reshape(Z.shape)
            case ScattererType.PENETRABLE | ScattererType.ABSORBING:
                pass

        ax.pcolormesh(X, Y, Z, shading="gouraud")
        match self.scatterer_type:
            case ScattererType.SOUND_HARD | ScattererType.SOUND_SOFT:
                for patch in self.scatterer_patchs:
                    ax.add_patch(patch())
            case ScattererType.PENETRABLE | ScattererType.ABSORBING:
                for patch in self.scatterer_patchs:
                    pass
                    # ax.add_patch(patch(penetrable=True))


        if show_edges: 
            lines = [ [E.P, E.Q] for E in self.Edges]
            ax.add_collection(LineCollection(lines, colors='k', linewidth=0.8))
        
        R = self.R
        H = self.H
        ax.axis('square')
        if self.half_infinite:
            ax.set_xlim([0,self.R])
        else:
            ax.set_xlim([-self.R,self.R])
        ax.set_ylim([0,H])
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$\\mathbf{\\hat{x}}$')
        


    def plot_mesh(self, ax=None):
        if self.meshed == False:
            print('cannot plot mesh before generating a mesh')
            return
        
        if ax is None:
            _, ax = plt.subplots( figsize=(15,3))
        lw = 0.8
        Lw = 2.

        for E in self.Edges:
            px, py = E.P
            qx, qy = E.Q
            match E.Type:
                case EdgeType.INNER:
                    ax.plot([px, qx], [py, qy], 'k', linewidth=lw)

                case EdgeType.GAMMA:
                    ax.plot([px, qx], [py, qy], 'g', linewidth=Lw)

                case EdgeType.SIGMA_L:
                    ax.plot([px, qx], [py, qy], '--r', linewidth=Lw)

                case EdgeType.SIGMA_R:
                    ax.plot([px, qx], [py, qy], '--r', linewidth=Lw)
    
                case EdgeType.D_OMEGA | EdgeType.COVER:
                    ax.plot([px, qx], [py, qy], '--b', linewidth=Lw)
                


        d = 0.2
        ax.axis('square')
        if self.half_infinite:
            ax.set_xlim([-d,self.R+d])
        else:
            ax.set_xlim([-self.R-d,self.R+d])
        ax.set_ylim([0-d,self.H+d])
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$\\mathbf{\\hat{x}}$')
        ax.set_yticks([0,self.H])
        ax.set_xticks([-self.R,0,self.R])



    def L2_norm(self,X, Y, Z):
        mask = self.in_scatterer(X.ravel(),Y.ravel())
        Ny, Nx = Z.shape
        L2 = np.sqrt( 2*self.R/Nx * self.H/Ny * np.sum(np.where(mask,0.,np.abs(Z.ravel())**2)))
        return L2





    

