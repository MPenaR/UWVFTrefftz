from DGTrefftz.two_dimensional_integrators import fek3_int
import numpy as np


TOL = 1E-6


def test_area():
    r_A = np.array([0, 0])
    r_B = np.array([1, 0])
    r_C = np.array([7, 1])

    I_exact = 0.5
    I_numeric = fek3_int(r_A=r_A, r_B=r_C, r_C=r_B,
                         f=lambda x, y: np.ones_like(x))

    message = f'{I_exact=}, {I_numeric=}'
    assert np.isclose(I_numeric, I_exact, TOL, TOL), message


def test_x():
    r_A = np.array([0, 0])
    r_B = np.array([1, 0])
    r_C = np.array([0, 1])

    I_exact = 1/6
    I_numeric = fek3_int(r_A=r_A, r_B=r_C, r_C=r_B, f=lambda x, y: y)

    message = f'{I_exact=}, {I_numeric=}'
    assert np.isclose(I_numeric, I_exact, TOL, TOL), message
