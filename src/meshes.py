'''Module for generating the different meshes.
 Creates a netgen mesh and returns the NG-solve wrapper.'''
from enum import Enum, auto
from netgen.geom2d import SplineGeometry
from ngsolve import Mesh

class ScattererType(Enum):
    '''Enumeration of the different scatterer types.'''
    PENETRABLE = auto()
    SOUND_SOFT = auto()
    SOUND_HARD = auto()






def waveguideMesh( h_max = 2., R = 10., H=1., c=(0,0), rad = (0.2), quad = False,
                   scatterer_type = ScattererType.PENETRABLE ):
    '''Mesh for the full waveguide.'''
    geo = SplineGeometry()
    geo.AddRectangle(p1=(-R,-H),
                    p2=( R, H),
                    bcs=["Gamma","Sigma_R","Gamma","Sigma_L"], #consider defining gamma_T and gamma_D
                    leftdomain=1,
                    rightdomain=0)
    match scatterer_type:
        case ScattererType.PENETRABLE:
            geo.AddCircle(c=c,
                        r=rad,
                        bc="D_Omega",
                        leftdomain=2,
                        rightdomain=1)
            geo.SetMaterial (1, "Omega_e")
            geo.SetMaterial (2, "Omega_i")
        case ScattererType.SOUND_SOFT | ScattererType.SOUND_HARD:
            geo.AddCircle(c=c,
                        r=rad,
                        bc="D_Omega",
                        leftdomain=0,
                        rightdomain=1)
            geo.SetMaterial (1, "Omega_e")


    Omega = Mesh(geo.GenerateMesh(maxh= h_max, quad_dominated=quad))
    
    return Omega

def testMesh(h_max = 2., quad=False, R = 10., H = 1.):
    '''Creates a simple mesh without scatterer for testing.'''
    geo = SplineGeometry()
    geo.AddRectangle(p1=(-R,-H),
                    p2=( R, H),
                    bcs=["Gamma","Sigma_R","Gamma","Sigma_L"],
                    leftdomain=1,
                    rightdomain=0)
    geo.SetMaterial (1, "Omega_e")

    Omega = Mesh(geo.GenerateMesh(maxh= h_max, quad_dominated=quad))
    
    return Omega


def toyMesh(H=1.):
    '''Creates a toy square mesh without scatterer for testing.'''
    geo = SplineGeometry()
    geo.AddRectangle(p1=(-2*H,-H),
                    p2=( 2*H, H),
                    bcs=["Gamma","Sigma_R","Gamma","Sigma_L"],
                    leftdomain=1,
                    rightdomain=0)
    geo.SetMaterial (1, "Omega_e")

    Omega = Mesh(geo.GenerateMesh(maxh= 2*H))
    
    return Omega

