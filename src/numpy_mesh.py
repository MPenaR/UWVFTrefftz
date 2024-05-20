'''module to implement what is going to be the basic "numpy mesh"'''
import numpy as np
from labels import EdgeType
# from enum import Enum, auto

# # not sure yet where to put this
# class EdgeType(Enum):
#     INNER = auto()
#     GAMMA = auto()
#     SIGMA_L = auto()
#     SIGMA_R = auto()
#     D_OMEGA = auto()
#     COVER = auto()


edge_dtype = np.dtype( [("midpoint", np.float64,2),
                        ("normal", np.float64, 2),
                        ("length", np.float64, 1),
                        ("label", EdgeType, 1)])

fun_dtype = np.dtype( [ ("direction", np.float64, 2), ("kappa", np.float64, 1)])
