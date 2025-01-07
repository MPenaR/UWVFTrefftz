"""
module for defining numpy array types
"""

from numpy.typing import NDArray
import numpy as np

real_array = NDArray[np.float64]
complex_array = NDArray[np.complex128]
int_array = NDArray[np.int64]
