# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: UWVFTrefftz--hy3v2Qt
#     language: python
#     name: python3
# ---

# %%
import numpy as np
from Trefft_tools import Assemble_blockMatrix, AssembleMatrix, TrefftzSpace
from domains import Waveguide, ScattererShape, ScattererType
import matplotlib.pyplot as plt

# %%
kappa = 8.
R = 0.5
H = 1.

Domain = Waveguide(R=R,H=H)


L = 0.5*H
Domain.add_scatterer(scatterer_shape=ScattererShape.CIRCLE, scatterer_type= ScattererType.ABSORBING, params=[(0,0.5*H), 0.8*L])
#Domain.add_scatterer(scatterer_shape=ScattererShape.RECTANGLE, scatterer_type= ScattererType.ABSORBING, params=[(0,0.5*H), L, L])

Domain.generate_mesh(h_max=H/1)
Domain.plot_mesh()

# %%
Nth = 15
n_e = 1
n_i = 4 + 16*1j
#n_i = -2 

V = TrefftzSpace(Domain=Domain, DOF_per_element=Nth, n= {"Omega_i" : n_i, "Omega_e" : n_e}, kappa=kappa, th0=0.)
V.absorbing


# %%
N_DtN = 15
G = Assemble_blockMatrix(V, Edges=Domain.Edges, H=H, k=kappa, N_p=Nth, a = 1/2, b = 1/2, d_1=1/2, d_2=1/2, N_DtN=N_DtN, th_0=0.)

# %%
A = AssembleMatrix( V, Edges=Domain.Edges, H=H, a = 1/2, b = 1/2, d_1 = 1/2, d_2 = 1/2, Np = N_DtN)

# %%
A_full = A.toarray()
G_full = G.toarray()
np.max(np.abs(A_full - G_full)) / np.max(np.abs(A_full))

# %%
plt.spy(A_full)

# %%
plt.spy(G_full)

# %%
plt.spy(A_full-G_full, precision=1E-14)
# plt.hlines(y=np.arange(0,V.N_DOF), xmin=0, xmax=V.N_DOF, linestyles='-',colors='r')
# plt.vlines(x=np.arange(0,V.N_DOF), ymin=0, ymax=V.N_DOF, linestyles='-',colors='r')

plt.hlines(y=np.arange(0,V.N_DOF,Nth), xmin=0, xmax=V.N_DOF, linestyles='--',colors='b')
plt.vlines(x=np.arange(0,V.N_DOF,Nth), ymin=0, ymax=V.N_DOF, linestyles='--',colors='b')


# %%
from domains import ScattererShape, ScattererType
kappa = 32.
R = 1
H = 1.

Domain = Waveguide(R=R,H=H)
Domain.add_scatterer(scatterer_shape=ScattererShape.CIRCLE, scatterer_type=ScattererType.SOUND_SOFT, params=[np.array([0,0.7*H]), 0.1*H])
Domain.generate_mesh(h_max=H/15)
Domain.plot_mesh()

# %%
Nth = 15

n_e = 1
n_i = 4 + 2*1j

th_0 = 0.

V = TrefftzSpace(Domain=Domain, DOF_per_element=Nth, n= {"Omega_i" : n_i, "Omega_e" : n_e}, kappa=kappa, th0=th_0)
N_DtN = 15
A = Assemble_blockMatrix(V, Edges=Domain.Edges, H=H, k=kappa, N_p=Nth, a = 1/2, b = 1/2, d_1=1/2, d_2=1/2, N_DtN=N_DtN, th_0=th_0 )

# %%
A = A.tocsr()

# %%
from Trefft_tools import AssembleGreenRHS_left, AssembleRHS

# b = AssembleGreenRHS_left(V, Edges=Domain.Edges, k=kappa, H=H,d_2=1/2, x_0=-1.5*R, y_0=0.3*H, M=N_DtN)
b = AssembleRHS(V=V, Edges=Domain.Edges, k=kappa, H=H, d_2=1/2, t=1)

# %%
from scipy.sparse.linalg import spsolve as solve 
p = solve(A, b, use_umfpack=True)

# %%
Ny = 300
Nx = int(R/H)*Ny
x = np.linspace(-R,R,Nx)
y = np.linspace(0,H, Ny)
X, Y = np.meshgrid(x,y)


from Trefft_tools import TrefftzFunction

f = TrefftzFunction(V, p)

U_tot = np.array([f(x,y) for (x,y) in zip(X.flatten(), Y.flatten())]).reshape(X.shape)

# %%
Domain.plot_field(X,Y,np.abs(U_tot))

# %%

# %%
