"""
module containing geometry entities and definitions.
"""
from numpy_types import real_array

from typing import NamedTuple
class Edge(NamedTuple):
    M : real_array
    N : real_array
    l : float 
    T : real_array
