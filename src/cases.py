from enum import Enum, auto

class Case(Enum):
    FUNDAMENTAL_SOLUTION = auto()
    PROPAGATING_MODE = auto()
    PENETRABLE_SCATTERER = auto()
    SOUND_SOFT_SCATTERER = auto()
    FINE_MESH_BARRIER = auto()