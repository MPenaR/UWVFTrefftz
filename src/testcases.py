from enum import Enum, auto

class TestCase(Enum):
    FUNDAMENTAL_CIRCLE = auto()
    PROPAGATING_MODE = auto()
    PENETRABLE_SCATTERER = auto()
    SOUND_SOFT_SCATTERER = auto()
    FINE_MESH_BARRIER = auto()
    FUNDAMENTAL_SQUARE = auto()
    SOUND_SOFT_SQUARE = auto()

class BoundaryCondiion(Enum): 
    DIRICHLET = auto()
    NEUMANN = auto()
    TRANSMISSION = auto()

