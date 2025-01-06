from two_dimensional_integrators import  fek3_int
import pytest 
import numpy as np 


TOL = 1E-6

# @pytest.mark.parametrize(('d_m', 'd_n'), directions )
def test_standard():
    r_A = np.array([0,0])
    r_B = np.array([1,0])
    r_C = np.array([0,1])

    I_exact = 0.5
    I_numeric = fek3_int(r_A=r_A, r_B=r_C, r_C=r_B, f=lambda x, y : np.ones_like(x))

    assert np.isclose( I_numeric, I_exact, TOL, TOL), f'{I_exact=}, {I_numeric=}'
