r"""
module for testing the block fluxes against the single fluxes"""

from single_fluxes import SoundHard
from block_fluxes import SoundHard_block
from geometry import Edge
import numpy as np
from FEM import TestFunction, TrialFunction

NTH = 15
TOL = 1E-14

def test_SoundHard_block():
    k = 8.0
    l = 1.
    M = np.array([1,1])
    T = np.array([1,0])
    N = np.array([0,1])
    d_1 = 0.5
    edge = Edge(l=l, M=M, N=N, T=T)

    th0 = 0
    th = np.linspace(0,2*np.pi, NTH, endpoint=False) + th0
    d = np.column_stack([np.cos(th), np.sin(th)])    
    d_d = np.transpose( np.stack( [np.subtract.outer(d[:,j],d[:,j]) for j in range(2)], axis=2 ), axes = [1,0,2])

    I_block = SoundHard_block(k=k, edge=edge, d=d, d_d=d_d, d_1=d_1)
    I = np.zeros((NTH,NTH),dtype=np.complex128)
    for i in range(NTH):
        for j in range(NTH):
            phi = TrialFunction(d=d[j], n=1)
            psi = TestFunction(d=d[i], n=1)
            I[i,j] = SoundHard( phi=phi, psi=psi, k=k, edge=edge, d_1=d_1)
    assert np.allclose(I,I_block,atol=TOL)
