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
from paper_plots import plot_hp_convergence
import numpy as np
from domains import Waveguide, ScattererShape, ScattererType
from FEM_solution import FEM_solution
from Trefft_tools import  TrefftzSpace, AssembleMatrix, AssembleRHS, Assemble_blockMatrix

from Trefft_tools import TrefftzFunction
#this should be a "solve system"
from scipy.sparse.linalg import spsolve 


# %%
kappa = 8.
R = 2*np.pi/kappa
H = 1.

c = (0,0.6*H)
Domain = Waveguide(R=R,H=H)

rad = 0.3*H
length = rad
width = rad

scatterer_shape = ScattererShape.RECTANGLE
scatterer_type = ScattererType.PENETRABLE

# Domain.add_scatterer( ScattererShape.CIRCLE, ScattererType.SOUND_SOFT, (c, rad))
Domain.add_scatterer( scatterer_shape=scatterer_shape, scatterer_type=scatterer_type, params=(c, rad, rad))
# Domain.add_fine_mesh_region(h_min=0.02)

Domain.generate_mesh(h_max=H/5)
Domain.plot_mesh()

# %%
Ny = 50
Nx = 10*Ny
x = np.linspace(-R,R,Nx)
y = np.linspace(0,H,Ny)
X, Y = np.meshgrid(x,y)
t = 0
N = 9 + 0j
N = 9

Z = FEM_solution( R=R, H=H, params={"c":c, "height" : length, "width" : width}, scatterer_shape=scatterer_shape, scatterer_type=scatterer_type, 
                 n=t,k_e=kappa,k_i= np.sqrt(N)*kappa, X=X, Y=Y, delta_PML=0.5*R, alpha=0.5*(4+2*1j))

# %%
Domain.plot_field(X, Y, np.abs(Z))

# %%
Nth = 15
th_0 = np.e/np.pi
th_0 = 0.
V = TrefftzSpace(Domain=Domain, DOF_per_element=Nth, kappa=kappa, n= {"Omega_i" : N, "Omega_e" : 1}, th0=th_0 )
N_modes = 15 #Number of modes for the DtN map
# "UWVF" parameters
a = 0.5
b = 0.5
d_1 = 0.5
d_2 = 0.5
A_old = AssembleMatrix(V=V, Edges=Domain.Edges, H=H, Np=N_modes, a=a, b=b, d_1=d_1, d_2=d_2)
A_block = Assemble_blockMatrix(V=V, Edges=Domain.Edges, th_0=th_0, H=H, k=kappa, N_p=Nth, a=a, b=b, d_1=d_1, d_2=d_2, N_DtN=N_modes)

NDOF = A_block.shape[0]
print(f'{NDOF} degrees of freedom.\n Matrix with {np.count_nonzero(A.toarray())} non-zero entries from a total of {NDOF**2}.\n "fullness" ratio: {np.count_nonzero(A.toarray())/NDOF**2 * 100 : .2f}%')

# %%
B = AssembleRHS(V, Domain.Edges, kappa, H, d_2=d_2, t = t)
from Trefft_tools import TrefftzFunction
#this should be a "solve system"
from scipy.sparse.linalg import spsolve 
A = A_block.tocsc()
DOFs = spsolve(A,B)
f = TrefftzFunction(V,DOFs)

# %%
u_Trefft =  np.reshape([ f(x_, y_) for x_, y_ in zip( X.ravel(), Y.ravel()) ], [Ny,Nx])


# %%
Domain.plot_field(X,Y,np.abs(u_Trefft), show_edges=True)


# %%
Domain.plot_field(X,Y,np.abs(u_Trefft - Z), show_edges=True)
print(f'The relative error is: {Domain.L2_norm(X,Y,np.abs(u_Trefft - Z))/Domain.L2_norm(X,Y,np.abs(Z)) : .2e}')

# %%
A_old = A_old.tocsc()
DOFs = spsolve(A_old,B)
f = TrefftzFunction(V,DOFs)

# %%
u_Trefft_old =  np.reshape([ f(x_, y_) for x_, y_ in zip( X.ravel(), Y.ravel()) ], [Ny,Nx])
Domain.plot_field(X,Y,np.abs(u_Trefft_old - Z), show_edges=True)
print(f'The relative error is: {Domain.L2_norm(X,Y,np.abs(u_Trefft_old - Z))/Domain.L2_norm(X,Y,np.abs(Z)) : .2e}')


# %%
Domain.plot_field(X,Y,np.abs(u_Trefft_old - u_Trefft), show_edges=True)
print(f'The relative error is: {Domain.L2_norm(X,Y,np.abs(u_Trefft_old - u_Trefft))/Domain.L2_norm(X,Y,np.abs(Z))}')


# %%
np.sum(np.abs(A - A_old))

# %%
kappa = 8.
R = 2*np.pi/kappa
H = 1.

Domain = Waveguide(R=R,H=H)
Domain.add_fine_mesh_region(h_min=0.02)

Domain.generate_mesh(h_max=H/5)
Domain.plot_mesh()

# %%
refinements = range(3,10,1)
N_ths = [3,5,7,9,11,13,15]


N_ref = len(refinements)
N_N_th = len(N_ths)

hs = np.zeros((N_ref), dtype=np.float64)
Hs = np.zeros((N_ref), dtype=np.float64)

conds = np.zeros((N_ref,N_N_th), dtype=np.float64)
errors = np.zeros((N_ref,N_N_th), dtype=np.float64)

factor = 8/8

kappa = factor*8.

R = factor*2*np.pi/kappa
H = 1.

Ny = 100
Nx = int(R/H*Ny)

x = np.linspace(-R,R,Nx)
y = np.linspace(0,H,Ny)
X, Y = np.meshgrid(x,y)

t = 1
beta = np.emath.sqrt(kappa**2 - (t*np.pi/H)**2)
u_exact = np.exp(1j*beta*X)*np.cos(t*np.pi*Y/H)


for (i,N) in enumerate(refinements):
    print(f"working on refinement: {N=}")
    h = H/N
    hs[i] = h
   
    Domain = Waveguide(R=R,H=H) 
    Domain.add_fine_mesh_region(factor = 0.9, h_min = 0.01)

    Domain.generate_mesh(h_max=h)

    side_ls = np.array([E.l for E in Domain.Edges])
    h_max = np.max(side_ls)
    Hs[i] = h_max
    print(f'{h_max=}')

    if i == 0:
        print('Initial refinement:')
        Domain.plot_mesh()

    for (j,Nth) in enumerate(N_ths):
        print(f"N_theta={Nth}...")
        th_0 = np.e/np.pi # no correct direction in the basis
        #th_0 = 0
        V = TrefftzSpace(Domain, Nth, kappa=kappa, n={"Omega_i" : 1 , "Omega_e" : 1}, th0=th_0 )
        N_modes = 15 #Number of modes for the DtN map
        # "UWVF" parameters
        a = 0.5
        b = 0.5
        d_1 = 0.5
        d_2 = 0.5

        A = Assemble_blockMatrix( V=V, Edges=Domain.Edges, H=H, k=kappa, N_p = Nth, th_0=th_0, a=a, b=b, d_1=d_1, d_2=d_2, N_DtN=N_modes)
        # B = AssembleGreenRHS_left(V = V, Edges=Domain.Edges, k=kappa_e, H=H, d_2=d_2, x_0 = c[0], y_0 = c[1], M = M)
        B = AssembleRHS(V=V, Edges=Domain.Edges, k=kappa, H=H, d_2=1/2, t=t)
        A = A.tocsc()
        
        DOFs = spsolve(A,B)
        f = TrefftzFunction(V,DOFs)

        u_Trefft =  np.reshape([ f(x_, y_) for x_, y_ in zip( X.ravel(), Y.ravel()) ], [Ny,Nx])
        errors[i,j] = Domain.L2_norm(X,Y,u_exact-u_Trefft)/Domain.L2_norm(X,Y,u_exact)


# %%
import matplotlib.pyplot as plt 

# %%
plot_hp_convergence(errors=errors, hs=hs, N_ths=N_ths, kappa_e=kappa, N_modes=N_modes, H=H)
#plt.savefig('fine_barrier_raw.png')

# %%
np.savez(file='fine_barrier_raw.npz', errors=errors, hs = hs, N_ths=N_ths)

# %%
from exact_solutions import GreenFunctionModes
from Trefft_tools import AssembleGreenRHS_left

refinements = range(6,13,1)
N_ths = [3,5,7,9,11,13,15]


N_ref = len(refinements)
N_N_th = len(N_ths)

hs = np.zeros((N_ref), dtype=np.float64)
Hs = np.zeros((N_ref), dtype=np.float64)

conds = np.zeros((N_ref,N_N_th), dtype=np.float64)
errors = np.zeros((N_ref,N_N_th), dtype=np.float64)

factor = 8/8

kappa_e = factor*8.
lambda_e = 2*np.pi/kappa_e
R = factor*lambda_e
H = 1.


c = (-1.5*R,0.3*H)


Ny = 100
Nx = int(R/H*Ny)

x = np.linspace(-R,R,Nx)
y = np.linspace(0,H,Ny)
X, Y = np.meshgrid(x,y)

M = 20
u_exact = GreenFunctionModes(k=kappa_e, H=H, XY=np.column_stack([X.ravel(), Y.ravel()]), x_0=c[0], y_0=c[1], M = M).reshape(X.shape)


for (i,N) in enumerate(refinements):
    print(f"working on refinement: {N=}")
    h = H/N
    hs[i] = h
   
    Domain = Waveguide(R=R,H=H) 
    Domain.generate_mesh(h_max=h)

    side_ls = np.array([E.l for E in Domain.Edges])
    h_max = np.max(side_ls)
    Hs[i] = h_max
    print(f'{h_max=}')

    if i == 0:
        print('Initial refinement:')
        Domain.plot_mesh()

    for (j,Nth) in enumerate(N_ths):
        print(f"N_theta={Nth}...")
        th_0 = 2*np.e/np.pi # no correct direction in the basis
        #th_0 = 0.
        V = TrefftzSpace(Domain, Nth, kappa=kappa_e, n={"Omega_i" : 1 , "Omega_e" : 1}, th0=th_0 )
        N_modes = 15 #Number of modes for the DtN map
        # "UWVF" parameters
        a = 0.5
        b = 0.5
        d_1 = 0.5
        d_2 = 0.5

        A = Assemble_blockMatrix( V=V, Edges=Domain.Edges, H=H, k=kappa_e, N_p = Nth, th_0=th_0, a=a, b=b, d_1=d_1, d_2=d_2, N_DtN=N_modes) 
        B = AssembleGreenRHS_left(V=V, Edges=Domain.Edges, k=kappa_e, H=H, d_2=d_2, x_0 = c[0], y_0=c[1], M=M)
        A = A.tocsc()
        
        DOFs = spsolve(A,B)
        f = TrefftzFunction(V,DOFs)

        u_Trefft =  np.reshape([ f(x_, y_) for x_, y_ in zip( X.ravel(), Y.ravel()) ], [Ny,Nx])
        errors[i,j] = Domain.L2_norm(X,Y,u_exact-u_Trefft)/Domain.L2_norm(X,Y,u_exact)


# %%
np.savez('green_new.npz', errors=errors, hs=hs, N_ths=N_ths)

# %%
import matplotlib.pyplot as plt
import numpy as np 
from paper_plots import plot_hp_convergence
from scipy.stats import linregress

def add_trend_line(IDs: list[int], xs, ys, errors, k, hs, ax):
    for ID, x, y in zip(IDs, xs, ys):
        result = linregress(np.log(k*hs), np.log(errors[:, ID]))
        m = result.slope
        n = result.intercept
        ax.plot(k*hs, 0.5*np.exp(n)*(k*hs)**m, '--k')
        ax.text(x,y,f'$\\propto (kh)^{{{m : .1f}}}$')


kappa_e = 8.
N_modes = 15
H = 1.

data = np.load('green_original.npz')
errors = data['errors']
hs = data['hs']
N_ths = data['N_ths']
ax = plot_hp_convergence(errors=errors, hs=hs, N_ths=N_ths, kappa_e=kappa_e, N_modes=N_modes, H=H)
add_trend_line(IDs=[2,5],xs = [1.,1.8], ys=[2E-3, 1E-7], errors=errors, k=kappa_e, hs=hs, ax = ax[1] )
plt.savefig('with_slopes_correct.pdf')

# %%
result = linregress(np.log(kappa_e*hs), np.log(errors[:,-1]))
m = result.slope
n = result.intercept
print(f'{m=} {n=}')

# %% [markdown]
# $$\log(E) = n + m \log(kh)$$
# $$E = e^n*(kh)^m$$
#

# %%
Domain.plot_field(X,Y,np.real(u_Trefft), show_edges=True)

# %% [markdown]
# # NtD Dependency

# %%
from exact_solutions import GreenFunctionModes
from Trefft_tools import AssembleGreenRHS_left

ks = [ 8, 12, 16, 20, 24, 28, 32]
ntds = np.arange(2,14)
N_ks = len(ks)
N_ntds = len(ntds)


errors = np.zeros((N_ks, N_ntds), dtype=np.float64)

for i, kappa_e in enumerate(ks):
    print(f'{kappa_e=}')    
    lambda_e = 2*np.pi/kappa_e
    R = 4*lambda_e
    H = 1.


    c = (-1.5*R,0.3*H)


    Ny = 100
    Nx = int(R/H*Ny)

    x = np.linspace(-R,R,Nx)
    y = np.linspace(0,H,Ny)
    X, Y = np.meshgrid(x,y)

    M = 20
    u_exact = GreenFunctionModes(k=kappa_e, H=H, XY=np.column_stack([X.ravel(), Y.ravel()]), x_0=c[0], y_0=c[1], M = M).reshape(X.shape)

    
    h = H/10

    Domain = Waveguide(R=R,H=H) 
    Domain.generate_mesh(h_max=h)


    for (j, N_DTN) in enumerate(ntds):
        print(f"{N_DTN=}")
        th_0 = 2*np.e/np.pi # no correct direction in the basis
        #th_0 = 0.
        N_th = 13
        V = TrefftzSpace(Domain, N_th, kappa=kappa_e, n={"Omega_i" : 1 , "Omega_e" : 1}, th0=th_0 )
        N_modes = N_DTN #Number of modes for the DtN map
        # "UWVF" parameters
        a = 0.5
        b = 0.5
        d_1 = 0.5
        d_2 = 0.5

        A = Assemble_blockMatrix( V=V, Edges=Domain.Edges, H=H, k=kappa_e, N_p = N_th, th_0=th_0, a=a, b=b, d_1=d_1, d_2=d_2, N_DtN=N_modes) 
        B = AssembleGreenRHS_left(V=V, Edges=Domain.Edges, k=kappa_e, H=H, d_2=d_2, x_0 = c[0], y_0=c[1], M=M)
        A = A.tocsc()
        
        DOFs = spsolve(A,B)
        f = TrefftzFunction(V,DOFs)

        u_Trefft =  np.reshape([ f(x_, y_) for x_, y_ in zip( X.ravel(), Y.ravel()) ], [Ny,Nx])
        errors[i,j] = Domain.L2_norm(X,Y,u_exact-u_Trefft)/Domain.L2_norm(X,Y,u_exact)


# %%
np.savez("ntd_dependency.npz", ntds = ntds, errors = errors, ks = ks )

# %%
import matplotlib.pyplot as plt 
inches_per_dot = 1/72.27
columnwidth = 370.38374 * inches_per_dot


figure_width = 0.7*columnwidth # cm
figure_height = 0.5*columnwidth # cm

import matplotlib

matplotlib.style.use({
    "font.size" : 7,
    "lines.linewidth" : 1.0,
    "lines.markersize":      3 

})

import numpy as np
data = np.load('ntd_dependency.npz')

ntds = data["ntds"]
errors = data["errors"]
ks = data["ks"]



fig, ax = plt.subplots(figsize=(figure_width, figure_height))


for i in range(N_ks):
    plt.semilogy(ntds,errors[i,:],'.-', label=f'$k={ks[i]}$, $\\lfloor \\frac{{k H}}{{\\pi}}\\rfloor = {int(ks[i]*1/np.pi)}$')
    #plt.semilogy(ntds,errors[i,:], label=f'$k={ks[i]}$, $M = {int(ks[i]*1/np.pi)}$')

    plt.legend()
    plt.ylabel('Relative $L_2$ error')
    plt.xlabel('Number of modes used for the approximation of the NtD map')

fig.subplots_adjust(left   = 0.15,
                    bottom = 0.15,
                    right  = 0.98,
                    top    = 0.98,
                    hspace = 0.35)


plt.savefig('NtD_dependency_correct.pdf')

# %%
for k in ks:
    print(f'{k=}')
    for i in range(0,15):
        print(f'k_hat = {i*np.pi : .2f}, {k**2 - (i*np.pi)**2 : .2f}')

# %%
