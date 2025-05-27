import numpy as np
from typing import Callable, TypeVar, Tuple

num = TypeVar("num", float, complex)

def fek3_int(f: Callable[[float, float], num],
             r_A: Tuple[float, float] = (0, 0),
             r_B: Tuple[float, float] = (1, 0),
             r_C: Tuple[float, float] = (0, 1)) -> num:
    """
    fek3_int:

    Numerically integrates on a triangle r_A, r_B, r_C
    the function f, using the set of 10 Fekete points
    """
    ABx = r_B[0] - r_A[0]
    ABy = r_B[1] - r_A[1]
    ACx = r_C[0] - r_A[0]
    ACy = r_C[1] - r_A[1]
    S = 0.5 * np.abs(ABx*ACy - ABy*ACx)
    J = S/2.
    points = np.array([[1./3, 1./3, 1./3],
                    [0., 0., 1.],
                    [0., 1., 0.],
                    [1., 0., 0.],
                    [0., 0.2763932023, 0.7236067977],
                    [0.7236067977, 0.2763932023, 0.],
                    [0.7236067977, 0., 0.2763932023],
                    [0., 0.7236067977, 0.2763932023],
                    [0.2763932023, 0.7236067977, 0.],
                    [0.2763932023, 0., 0.7236067977]]) 
    
    x = r_A[0]*points[:, 0] + r_B[0]*points[:, 1] + r_C[0]*points[:, 2]
    y = r_A[1]*points[:, 0] + r_B[1]*points[:, 1] + r_C[1]*points[:, 2]
    
    w = np.array([0.9, 0.1/3, 0.1/3, 0.1/3, 1./6, 1./6, 1./6, 1./6, 1./6, 1./6])
    z = f(x, y)
    I = np.sum(z*w)
    return J*I 
