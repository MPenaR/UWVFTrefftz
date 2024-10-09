from enum import Enum, auto

class EdgeType(Enum):
    INNER = auto()
    GAMMA = auto()
    SIGMA_L = auto()
    SIGMA_R = auto()
    D_OMEGA = auto()
    COVER = auto()
    # IN_OMEGA = auto()
