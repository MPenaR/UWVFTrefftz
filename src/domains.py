# module for the different domains

from netgen.geom2d import SplineGeometry
from ngsolve import Mesh
from enum import Enum, auto
from geometry_tools import Edge


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
    def __init__(self, R = 10., H = 2.):
        self.R = R 
        self.H = H
        self.geo = SplineGeometry()
        self.geo.AddRectangle(p1=(-R,0), p2=( R, H), bcs=["Gamma","Sigma_R","Gamma","Sigma_L"], leftdomain=1, rightdomain=0)
        self.geo.SetMaterial (1, "Omega_e")
        self.scatterer_type = None

    def add_scatterer( self, scatterer_shape : ScattererShape, scatterer_type : ScattererType, params : list) :
        self.scatterer_type = scatterer_type
        match scatterer_shape:
            case ScattererShape.CIRCLE:
                c, r = params
                if scatterer_type == ScattererType.PENETRABLE:
                    self.geo.AddCircle(c=c, r=r, leftdomain=2, rightdomain=1)
                    self.geo.SetMaterial (2, "Omega_i")
                else:
                    self.geo.AddCircle(c=c, r=r, bc="D_Omega", leftdomain=0, rightdomain=1)
            case ScattererShape.RECTANGLE:
                c, length, height = params
                if scatterer_type == ScattererType.PENETRABLE:
                    self.geo.AddRectangle(p1=(c[0]-length/2,c[1]-height/2), p2=(c[0]+length/2,c[1]+height/2), leftdomain=2, rightdomain=1)
                    self.geo.SetMaterial (2, "Omega_i")
                else:
                    self.geo.AddRectangle(p1=(c[0]-length/2,c[1]-height/2), p2=(c[0]+length/2,c[1]+height/2), leftdomain=0, rightdomain=1)
                    
    def generate_mesh(self, h_max = 1.):
        Omega = Mesh(self.geo.GenerateMesh(maxh= h_max))
        match self.scatterer_type:
            case None:
                Edges = [ Edge(Omega, e, None)  for e in Omega.edges ]
            case _:
                Edges = [ Edge(Omega, e, (0,1.))  for e in Omega.edges ] # HARDCODED
        return Omega, Edges
    

