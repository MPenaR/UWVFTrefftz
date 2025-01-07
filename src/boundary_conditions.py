"""
Module for defining boundary conditions.
"""
from enum import Enum, auto

# class BoundaryCondition(Enum): 
#     DIRICHLET = 1
#     SOUND_SOFT = 1
#     NEUMANN = 2
#     SOUND_HARD = 2
#     TRANSMISSION = 3

class BoundaryCondition(Enum):
    '''
    Types of boundary condition. (transmission conditions are not "boundary" conditions)
    '''
    DIRICHLET = auto()
    NEUMANN = auto()
    RADIATING = auto()

