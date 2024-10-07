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
from paper_plots import plot_hp_convergence
import matplotlib.pyplot as plt 

# %%
data = np.load('fundamental_outside8.npz')
errors = data["errors"]
hs = data["hs"]
N_ps = data["N_ths"]
kappa_e  = 8

# %%

plot_hp_convergence(errors,hs,N_ps,kappa_e,N_modes=15,H=1)

# %%
from matplotlib import ticker as mticker

import numpy as np


inches_per_dot = 1/72.27
cm2inch = 1/2.54 # inch per cm
columnwidth = 630.185 * inches_per_dot
columnwidth = 469.75502 * inches_per_dot
#columnwidth = 524.07272*inches_per_dot

columnwidth=columnwidth*0.9

left_margin = 3. * cm2inch # cm
right_margin = 1.*cm2inch  # cm
figure_width = columnwidth # cm
figure_height = columnwidth/1. # cm
top_margin = 2.*cm2inch    # cm
bottom_margin = 1.5*cm2inch # cm

box_width = left_margin + figure_width + right_margin   # cm
box_height = top_margin + figure_height + bottom_margin # cm

fig, ax = plt.subplots(nrows=2,figsize=(box_width,box_height))

for err, h in zip(errors,hs):
    ax[0].semilogy(N_ps,err,'.-', label=f'$h_\\mathrm{{max}} = {h: .1e}$')

ax[0].set_xlabel('$N_p$')
ax[0].set_ylabel('$\\left\\Vert u - u_h\\right\\Vert_2^2 \\, / \\, \\left\\Vert u \\right\\Vert_2^2$')
ax[0].yaxis.set_major_locator(mticker.LogLocator(numticks=999))
ax[0].yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
ax[0].set_xticks(range(3,17,2))
ax[0].grid(True,which="major",ls='--')
ax[0].legend()


for err, N_th in zip(errors.transpose(),N_ps):
    ax[1].loglog(kappa_e*hs,err,'.-', label=f'$N_P = {N_th}$')

ax[1].set_xlabel('$\\kappa h$')
ax[1].set_ylabel('$\\left\\Vert u - u_h\\right\\Vert_2^2 \\, / \\, \\left\\Vert u \\right\\Vert_2^2$')
ax[1].legend(loc="lower right")
ax[1].yaxis.set_major_locator(mticker.LogLocator(numticks=999))
ax[1].yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))

plt.grid(True,which="major",ls='--')


fig.subplots_adjust(left   = left_margin / box_width,
                    bottom = bottom_margin / box_height,
                    right  = 1. - right_margin / box_width,
                    top    = 1. - top_margin   / box_height)

plt.loglog(kappa_e*hs, np.e**(-5)*(kappa_e*hs)**(1.5),'--')
plt.text(x=1., y=0.001, s=r'$y\propto(\kappa h)^{\frac{3}{2}}$')



# plt.loglog(kappa_e*hs, np.e**(-7)*(kappa_e*hs)**(4.15),'--')
# plt.text(x=1., y=0.004, s=r'$y\propto(\kappa h)^{4.15}$')

# plt.loglog(kappa_e*hs, np.e**(-19)*(kappa_e*hs)**(6.7),'--')
# plt.text(x=1.5, y=1E-8, s=r'$y\propto(\kappa h)^{6.7}$')

#plt.savefig('with_1.5_line.pdf')

# %%
from scipy.stats import linregress

results = linregress(np.log(kappa_e*hs),np.log(errors[:,2]))
results

# %% [markdown]
# $$
# \log(\mathrm{err})\approx  -8.5 + 4.15*\log(\kappa h)
# $$
#
# $$
# \mathrm{err}\approx  e^{-8.5}\left(\kappa h\right)^{4.15}
# $$

# %%
np.e**(-8.5)

# %%
results = linregress(np.log(kappa_e*hs),np.log(errors[:,0]))
results


# %%
results = linregress(np.log(kappa_e*hs),np.log(errors[:,5]))
results


# %%
results = linregress(np.log(kappa_e*hs),np.log(errors[:,0]))
results


# %%
results = linregress(np.log(kappa_e*hs),np.log(errors[:,1]))
results


# %%
from exact_solutions import GreenFunctionModes
from FEM_solution import FEM_solution
from domains import Waveguide, ScattererShape, ScattererType
from Trefft_tools import TrefftzFunction, TrefftzSpace, Assemble_blockMatrix, AssembleGreenRHS_left, AssembleRHS

from scipy.sparse.linalg import spsolve as solve




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

kappa_e = factor*8.
lambda_e = 2*np.pi/kappa_e
R = factor*lambda_e
H = 1.


c = (-1.5*R,0.3*H)

#t=1

Ny = 100
Nx = int(R/H*Ny)

x = np.linspace(-R,R,Nx)
y = np.linspace(0,H,Ny)
X, Y = np.meshgrid(x,y)

# M = 32
# u_exact = np.reshape( GreenFunctionModes(kappa_e, H, np.stack([X.ravel(),Y.ravel()], axis=1), c[0], c[1], M=M), (Ny,Nx))
center = (0, 0.6*H)
t = 1
u_exact = FEM_solution(R, H, params={"c" : center, "width" : np.sqrt(2)*0.1*H, "height" : np.sqrt(2)*0.1*H},
             scatterer_shape=ScattererShape.RECTANGLE, scatterer_type=ScattererType.SOUND_SOFT,
             n = t, k_e=kappa_e, k_i=kappa_e, X=X, Y=Y, delta_PML=0.5*R, alpha=0.5*(4+2*1j) )

# u_exact = FEM_solution(R, H, params={"c" : center, "rad" : np.sqrt(2)*0.1*H},
#              scatterer_shape=ScattererShape.CIRCLE, scatterer_type=ScattererType.SOUND_SOFT,
#              n = t, k_e=kappa_e, k_i=kappa_e, X=X, Y=Y, delta_PML=0.5*R, alpha=0.5*(4+2*1j) )



for (i,N) in enumerate(refinements):
    print(f"working on refinement: {N=}")
    h = H/N
    hs[i] = h
   
    Domain = Waveguide(R=R,H=H) 
    # Domain.add_fine_mesh_region(factor = 0.9, h_min = 0.01)
    Domain.add_scatterer( ScattererShape.RECTANGLE, ScattererType.SOUND_SOFT, (center, np.sqrt(2)*0.1*H, np.sqrt(2)*0.1*H))
    # Domain.add_scatterer( ScattererShape.CIRCLE, ScattererType.SOUND_SOFT, (center, np.sqrt(2)*0.1*H))

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
        V = TrefftzSpace(Domain, Nth, kappa=kappa_e, n={"Omega_i" : 1 , "Omega_e" : 1}, th0=th_0 )
        N_modes = 15 #Number of modes for the DtN map
        # "UWVF" parameters
        a = 0.5
        b = 0.5
        d_1 = 0.5
        d_2 = 0.5

        A = Assemble_blockMatrix( V=V, Edges=Domain.Edges, H=H, k=kappa_e, N_p = Nth, th_0=th_0, a=a, b=b, d_1=d_1, d_2=d_2, N_DtN=N_modes)
        # B = AssembleGreenRHS_left(V = V, Edges=Domain.Edges, k=kappa_e, H=H, d_2=d_2, x_0 = c[0], y_0 = c[1], M = M)
        B = AssembleRHS(V=V, Edges=Domain.Edges, k=kappa_e, H=H, d_2=1/2, t=t)
        A = A.tocsc()
        
        DOFs = solve(A,B)
        f = TrefftzFunction(V,DOFs)

        u_Trefft =  np.reshape([ f(x_, y_) for x_, y_ in zip( X.ravel(), Y.ravel()) ], [Ny,Nx])
        errors[i,j] = Domain.L2_norm(X,Y,u_exact-u_Trefft)/Domain.L2_norm(X,Y,u_exact)


# %%
plot_hp_convergence(errors=errors, hs=hs, N_ths=N_ths, kappa_e=kappa_e, N_modes=N_modes, H=H)
# plt.loglog(kappa_e*hs, np.e**(-5)*(kappa_e*hs)**(1.5),'--')
# plt.text(x=1., y=0.001, s=r'$y\propto(\kappa h)^{\frac{3}{2}}$')
plt.savefig('soundsoft_square_mode_1_k_8.pdf')


# %%
Domain.plot_field(X,Y, np.real(u_exact))

# %%
Domain.plot_field(X,Y, np.real(u_Trefft))

# %%
Domain.plot_mesh()

# %%
Domain.plot_field(X,Y, np.abs(u_Trefft-u_exact))

# %%
