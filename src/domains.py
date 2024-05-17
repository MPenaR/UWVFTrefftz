# module for the different domains

from netgen.geom2d import SplineGeometry
from ngsolve import Mesh
from enum import Enum, auto
from geometry_tools import Edge, EdgeType
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection

class ScattererType(Enum):
    '''Enumeration of the different scatterer types.'''
    PENETRABLE = auto()
    SOUND_SOFT = auto()
    SOUND_HARD = auto()

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

    def add_scatterer( self, scatterer_shape : ScattererShape, scatterer_type : ScattererType, params : list) :
        self.scatterer_type = scatterer_type
        kwargs = {"edgecolor" : "k", "facecolor" : "grey", "linewidth" : 4}
        match scatterer_shape:
            case ScattererShape.CIRCLE:
                c, r = params

                self.scatterer_markers.append(lambda  x, y, c=c, r=r: (x-c[0])**2 + (y-c[1])**2 <= r**2)
                self.scatterer_patchs.append( lambda c=c, r=r, kwargs=kwargs : Circle(xy=c, radius=r, **kwargs))


                if scatterer_type == ScattererType.PENETRABLE:
                    self.geo.AddCircle(c=c, r=r, leftdomain=2, rightdomain=1)
                    self.geo.SetMaterial (2, "Omega_i")
                else:
                    self.geo.AddCircle(c=c, r=r, bc="D_Omega", leftdomain=0, rightdomain=1)
            case ScattererShape.RECTANGLE:
                c, width, height = params

                self.scatterer_markers.append(lambda x, y, c=c, width=width, height=height: np.logical_and( np.abs(x - c[0])<= width/2, np.abs(y - c[1])<= height/2 ))
                self.scatterer_patchs.append( lambda c=c, width=width, height=height, kwargs=kwargs :Rectangle(xy=(c[0] - width/2, c[1]-height/2), height=height, width=width, **kwargs))

                if scatterer_type == ScattererType.PENETRABLE:
                    self.geo.AddRectangle(p1=(c[0]-width/2,c[1]-height/2), p2=(c[0]+width/2,c[1]+height/2), leftdomain=2, rightdomain=1)
                    self.geo.SetMaterial (2, "Omega_i")
                else:
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
        
        # return self.Omega, self.Edges
        return  
    def plot_field(self, X, Y, Z, show_edges = False, ax = None):
        if ax is None: 
            _, ax = plt.subplots( figsize=(15,3))
        mask = self.in_scatterer(X.ravel(),Y.ravel())
        Z = np.where(mask,np.nan,Z.ravel()).reshape(Z.shape)
        ax.pcolormesh(X, Y, Z, shading="gouraud")
        for patch in self.scatterer_patchs:
            ax.add_patch(patch())


        if show_edges: 
            lines = [ [E.P, E.Q] for E in self.Edges]
            ax.add_collection(LineCollection(lines, colors='k'))
        
        R = self.R
        H = self.H
        ax.axis('square')
        if self.half_infinite:
            ax.set_xlim([0,self.R])
        else:
            ax.set_xlim([-self.R,self.R])
        ax.set_ylim([0,H])

    # match self.scatterer_type:
    #     case ScattererType.PENETRABLE:
    #         ax.set_title(f'Penetrable: mode number: {t}, {Nth} plane waves per triangle, condition number = $\\mathtt{{ {Ncond: .3f} }}$, $\\kappa={kappa_e:.1f}$')
    #     case ScattererType.SOUND_SOFT:
    #         ax.set_title(f'Sound-soft: mode number: {t}, {Nth} plane waves per triangle, condition number = $\\mathtt{{ {Ncond: .3f} }}$, $\\kappa={kappa_e:.1f}$')
    #     case ScattererType.GREEN_FUNC:
    #         ax.set_title(f'Green function case, {Nth} plane waves per triangle, condition number =   $\\mathtt{{ {Ncond: .3f} }}$, $\\kappa={kappa_e:.1f}$')


    # if scatterer_type != ScattererType.NONE:
    #     ax.add_patch(scatterer())

    def plot_mesh(self, ax=None):
        if ax is None:
            _, ax = plt.subplots( figsize=(15,3))
        lw = 2

        for E in self.Edges:
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
    
                case EdgeType.D_OMEGA | EdgeType.COVER:
                    ax.plot([px, qx], [py, qy], '--b', linewidth=lw)
                


        d = 0.2
        ax.axis('square')
        if self.half_infinite:
            ax.set_xlim([-d,self.R+d])
        else:
            ax.set_xlim([-self.R-d,self.R+d])
        ax.set_ylim([0-d,self.H+d])

    def L2_norm(self,X, Y, Z):
        mask = self.in_scatterer(X.ravel(),Y.ravel())
        Ny, Nx = Z.shape
        L2 = np.sqrt( 2*self.R/Nx * self.H/Ny * np.sum(np.where(mask,0.,np.abs(Z.ravel())**2)))
        return L2





    

