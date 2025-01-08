"""
module containing geometry entities and definitions.
"""
from numpy_types import real_array
from numpy import array
from numpy.linalg import norm
from typing import NamedTuple
class Edge(NamedTuple):
    M : real_array
    N : real_array
    l : float 
    T : real_array


def midpoint(P : real_array, Q : real_array) -> real_array:
    """Computes the midpoint of the segment PQ:
    
    Parameters
    ----------

    - P : real_array
    - Q : real_array
    
    Returns
    -------
    - M : real_array

    """
    return 0.5*(P + Q)

def tangent(P : real_array, Q : real_array) -> real_array:
    """Computes the unitary tangent vector of the segment PQ:
    
    Parameters
    ----------

    - P : real_array
    - Q : real_array
    
    Returns
    -------
    - T : real_array

    """
    return(Q - P)/norm(Q-P)

def normal(P : real_array, Q : real_array) -> real_array:
    r"""Computes the unitary normal vector of the segment PQ:
    The normal is chosen such that: 

    .. math::
        \boldsymbol{\tau} \times \mathbf{n} = \mathbf{k}
    
    Parameters
    ----------

    - P : real_array
    
    - Q : real_array
    
    Returns
    -------
    - N : real_array

    """

    T = tangent(P,Q)

    return array([-T[1],T[0]])

def length(P : real_array, Q : real_array) -> float:
    """Computes the length of the segment PQ:
    
    Parameters
    ----------

    - P : real_array
    - Q : real_array
    
    Returns
    -------
    - l : float

    """
    return norm(Q - P)