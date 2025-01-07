"""
Module for FEM related definitions and functions"""

from typing import NamedTuple
from numpy_types import real_array

class TestFunction(NamedTuple):
    d : real_array
    n : float

class TrialFunction(NamedTuple):
    d : real_array
    n : float
