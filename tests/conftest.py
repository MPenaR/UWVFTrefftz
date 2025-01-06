import numpy as np
import pytest
from itertools import product


@pytest.fixture
def directions(NTH = 3):
    return list(product([(np.cos(th), np.sin(th)) for th in np.linspace(0, np.pi/2, NTH, endpoint=False)],
                        [(np.cos(th), np.sin(th)) for th in np.linspace(0, np.pi/2, NTH, endpoint=False)]))



