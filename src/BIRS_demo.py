# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: UWVFTrefftz--hy3v2Qt
#     language: python
#     name: python3
# ---

# +
import matplotlib.pyplot as plt

from paper_plots import plot_hp_convergence


import numpy as np

from testcases import TestCase

from domains import Waveguide, ScattererShape, ScattererType
# -

# # Sandbox

# +
kappa_e = 8.
lambda_e = 2*np.pi/kappa_e
R = lambda_e
H = 1.

half_infinite = False
if half_infinite:
    Domain = Waveguide(R=2*R,H=H, half_infinite=half_infinite)
    c = (-0.2*R,0.1*H)
else:
    c = (0,0.7*H)
    Domain = Waveguide(R=R,H=H)
rad = 0.4*H

rad = 0.1*H


scatterer_shape = ScattererShape.RECTANGLE
scatterer_type = ScattererType.PENETRABLE

#Domain.add_scatterer( ScattererShape.CIRCLE, ScattererType.SOUND_SOFT, (c, rad))
Domain.add_scatterer( scatterer_shape=scatterer_shape, scatterer_type=scatterer_type, params=(c, 2*rad, 2*rad))

Domain.generate_mesh(h_max=H/8)
Domain.plot_mesh()

# +
from FEM_solution import FEM_solution

Ny = 50
Nx = 10*Ny
x = np.linspace(-R,R,Nx)
y = np.linspace(0,H,Ny)
X, Y = np.meshgrid(x,y)
t = 0
kappa_e = 8.0
N = 2
kappa_i = np.sqrt(N)*kappa_e

Z = FEM_solution( R=R, H=H, params={"c":c, "height" : 2*rad, "width" : 2*rad}, scatterer_shape=scatterer_shape, scatterer_type=scatterer_type, 
                 n=t,k_e=kappa_e,k_i=kappa_i, X=X, Y=Y)
# -

Domain.plot_field(X,Y,np.abs(Z))

# +
from Trefft_tools import  TrefftzSpace
Nth = 15
th_0 = np.e/np.pi # no correct direction in the basis

#th_0= 0. # right direction in the basis
V = TrefftzSpace(Domain.Omega, Nth, {"Omega_i" : kappa_i , "Omega_e" : kappa_e}, th_0 )

# -

from Trefft_tools import AssembleMatrix
N_modes = 15 #Number of modes for the DtN map
# "UWVF" parameters
a = 0.5
b = 0.5
d_1 = 0.5
d_2 = 0.5
A = AssembleMatrix(V, Domain.Edges, H, Np=N_modes, a=a, b=b, d_1=d_1, d_2=d_2)
NDOF = A.shape[0]
print(f'{NDOF} degrees of freedom.\n Matrix with {np.count_nonzero(A.toarray())} non-zero entries from a total of {NDOF**2}.\n "fullness" ratio: {np.count_nonzero(A.toarray())/NDOF**2 * 100 : .2f}%')

from checking_tools import plot_sparsity
plot_sparsity(A)

# +
# Ncond = np.linalg.cond(A.toarray())
# Ncond
# -

from Trefft_tools import AssembleGreenRHS, AssembleRHS


B = AssembleRHS(V, Domain.Edges, kappa_e, H, d_2=d_2, t = t)

from Trefft_tools import TrefftzFunction
#this should be a "solve system"
from scipy.sparse.linalg import spsolve 
A = A.tocsc()
DOFs = spsolve(A,B)
f = TrefftzFunction(V,DOFs)

plt.plot(np.abs(DOFs),'.')

# +
Ny = 50
Nx = 10*Ny
if half_infinite:
    x = np.linspace(0,2*R,Nx)
else:
    x = np.linspace(-R,R,Nx)
y = np.linspace(0,H,Ny)
X, Y = np.meshgrid(x,y)

# Z = np.reshape(f(X.ravel(), Y.ravel()), [Ny,Nx]) FIX THIS, EVALUATION SHOULD BE VECTORIZED

u_Trefft =  np.reshape([ f(x_, y_) for x_, y_ in zip( X.ravel(), Y.ravel()) ], [Ny,Nx])
# u_exact = np.exp(1j*np.emath.sqrt(kappa_e**2 - (t*np.pi/H)**2)*X)*np.cos(t*np.pi*Y/H)
u_exact = Z = FEM_solution( R=R, H=H, params={"c":c, "height" : 2*rad, "width" : 2*rad}, scatterer_shape=scatterer_shape, scatterer_type=scatterer_type, n=t,k_e=kappa_e,k_i=kappa_i, X=X, Y=Y)

# +

Domain.plot_field(X,Y,np.real(u_Trefft), show_edges=True)

# +
from FEM_solution import FEM_solution

# u_exact = FEM_solution(R=R,H=H, params={"c":c, "rad":rad}, scatterer_shape=scatterer_shape, scatterer_type=scatterer_type, n=t, k_e=kappa_e, k_i=kappa_i,
#                        X=X, Y=Y)

u_exact = FEM_solution(R=R,H=H, params={"c":c, "width": rad, "height" : rad }, scatterer_shape=scatterer_shape, scatterer_type=scatterer_type, n=t, k_e=kappa_e, k_i=kappa_i,
                       X=X, Y=Y)


Domain.plot_field(X,Y,np.real(u_exact))


# +
fig, ax = plt.subplots(ncols=3, figsize=(18,4))
Domain.plot_field(X,Y,np.real(u_Trefft),ax=ax[0], show_edges=True)
Domain.plot_field( X,Y,np.real(u_exact),ax=ax[1], show_edges=False)
Domain.plot_field( X,Y,np.abs(u_Trefft-u_exact),ax=ax[2], show_edges=True)
ax[0].set_title('$\\mathrm{Re}(u_\\mathrm{Trefftz})$')
ax[1].set_title('$\\mathrm{Re}(u_\\mathrm{exact})$')
ax[2].set_title('$\\left|u_\\mathrm{exact}-u_\\mathrm{Trefftz}\\right|$')

fig.suptitle(f'$L_2$ error: {Domain.L2_norm(X,Y,u_exact-u_Trefft)/Domain.L2_norm(X,Y,u_exact) : .2e}')

#plt.savefig(f'error_circle_k_{kappa_e:.3f}.png')
# -

du_Trefft_dy =  np.reshape([ f.dy(x_, y_) for x_, y_ in zip( X.ravel(), Y.ravel()) ], [Ny,Nx])
fig, ax = plt.subplots(ncols=2,figsize=(16,4))
Domain.plot_field(X,Y,np.abs(du_Trefft_dy), ax=ax[0])
ax[1].plot(y,np.real(du_Trefft_dy)[:,36])

plt.semilogy(x,np.abs(du_Trefft_dy[0,:]))
plt.semilogy(x,np.abs(du_Trefft_dy[-1,:]))
# plt.plot(x,np.real(du_Trefft_dy[0,:]))
# plt.plot(x,np.real(du_Trefft_dy[-1,:]))

plt.plot(np.real(u_exact - u_Trefft)[:,0])

# # Fundamental solution outside

# +
kappa_e = 8.
lambda_e = 2*np.pi/kappa_e
R = lambda_e
H = 1.

half_infinite = False
if half_infinite:
    Domain = Waveguide(R=2*R,H=H, half_infinite=half_infinite)
    c = (-0.2*R,0.1*H)
else:
    c = (0,0.7*H)
    Domain = Waveguide(R=R,H=H)

rad = 0.1*H

Domain.generate_mesh(h_max=H/10)
Domain.plot_mesh()

# +
Ny = 50
Nx = 10*Ny
x = np.linspace(-R,R,Nx)
y = np.linspace(0,H,Ny)
X, Y = np.meshgrid(x,y)


c = np.array([-R-0.5,H/3])

u_exact = np.reshape( GreenFunctionModes(np.real(kappa_e),Domain.H, np.stack([X.ravel(),Y.ravel()], axis=1), c[0], c[1], M=100), (Ny,Nx))

Domain.plot_field(X,Y,np.abs(u_exact),show_edges=False,ax=None,source=c)

# +
Nth = 15
th_0 = np.e/np.pi # no prefered direction in the basis
#th_0= 0. # prefered direction in the basis
V = TrefftzSpace(Domain.Omega, Nth, {"Omega_i" : kappa_e , "Omega_e" : kappa_e}, th_0 )

N_modes = 15 #Number of modes for the DtN map
# "UWVF" parameters
a = 0.5
b = 0.5
d_1 = 0.5
d_2 = 0.5
A = AssembleMatrix(V, Domain.Edges, H, Np=N_modes, a=a, b=b, d_1=d_1, d_2=d_2)
NDOF = A.shape[0]
print(f'{NDOF} degrees of freedom.\n Matrix with {np.count_nonzero(A.toarray())} non-zero entries from a total of {NDOF**2}.\n "fullness" ratio: {np.count_nonzero(A.toarray())/NDOF**2 * 100 : .2f}%')


# -

from Trefft_tools import AssembleGreenRHS_left
B = AssembleGreenRHS_left(V, Domain.Edges, kappa_e, H, d_2=1/2, x_0= c[0], y_0=c[1], M=40)

A = A.tocsc()
DOFs = spsolve(A,B)
f = TrefftzFunction(V,DOFs)

u_Trefft =  np.reshape([ f(x_, y_) for x_, y_ in zip( X.ravel(), Y.ravel()) ], [Ny,Nx])
Domain.plot_field(X,Y,np.abs(u_Trefft),show_edges=True,ax=None,source=c)

# +
fig, ax = plt.subplots(ncols=3, figsize=(12,4), sharey=True)
Domain.plot_field(X,Y,np.real(u_Trefft),ax=ax[0], show_edges=True)
Domain.plot_field( X,Y,np.real(u_exact),ax=ax[1], show_edges=False)
Domain.plot_field( X,Y,np.abs(u_Trefft-u_exact),ax=ax[2], show_edges=True)

ax[0].set_title('$\\mathrm{Re}(u_\\mathrm{Trefftz})$')
ax[1].set_title('$\\mathrm{Re}(u_\\mathrm{exact})$')
ax[2].set_title('$\\left|u_\\mathrm{exact}-u_\\mathrm{Trefftz}\\right|$')
for i in range(1,3):
    ax[i].set_ylabel("")

fig.suptitle(f'Relative $L_2$ error: {Domain.L2_norm(X,Y,u_exact-u_Trefft)/Domain.L2_norm(X,Y,u_exact) : .2e}')

# -

# # number of propagating modes
#
# maximum $t$ such that $\left(\kappa H\right)^2 \ge \left(t \pi \right)^2 $, that is:
#
# $$
# t = \left\lfloor\frac{\kappa H}{\pi}\right\rfloor
# $$

k = np.linspace(4,64,100)
t = np.floor(k*H/np.pi)
plt.plot(k,t)

# +
refinements = range(3,10,1)
N_ths = [3,5,7,9,11,13,15]


N_ref = len(refinements)
N_N_th = len(N_ths)

hs = np.zeros((N_ref), dtype=np.float64)
conds = np.zeros((N_ref,N_N_th), dtype=np.float64)
errors = np.zeros((N_ref,N_N_th), dtype=np.float64)

factor = 8/8

kappa_e = factor*8.
lambda_e = 2*np.pi/kappa_e
R = factor*lambda_e
H = 1.


Ny = 100
Nx = int(R/H*Ny)
x = np.linspace(-R,R,Nx)
y = np.linspace(0,H,Ny)
X, Y = np.meshgrid(x,y)

M = 32
c = np.array([-R-0.5,H/3])
u_exact = np.reshape( GreenFunctionModes(np.real(kappa_e),Domain.H, np.stack([X.ravel(),Y.ravel()], axis=1), c[0], c[1], M=M), (Ny,Nx))


for (i,N) in enumerate(refinements):
    print(f"working on refinement: {N=}")
    h = H/(N)
    hs[i] = h
    Domain = Waveguide(R=R,H=H) 
    Domain.generate_mesh(h_max=h)

    side_ls = np.array([E.l for E in Domain.Edges])
    h_max = np.max(side_ls)
    print(f'{h_max=}')

    for (j,Nth) in enumerate(N_ths):
        print(f"N_theta={Nth}...")
        th_0 = np.e/np.pi # no correct direction in the basis
        V = TrefftzSpace(Domain.Omega, Nth, {"Omega_i" : kappa_e , "Omega_e" : kappa_e}, th_0 )
        N_modes = 15 #Number of modes for the DtN map
        # "UWVF" parameters
        a = 0.5
        b = 0.5
        d_1 = 0.5
        d_2 = 0.5

        A = AssembleMatrix(V, Domain.Edges, H, Np=N_modes, a=a, b=b, d_1=d_1, d_2=d_2)
        #_, _, conds[i,j] = cond(A)
        B = AssembleGreenRHS_left(V, Domain.Edges, kappa_e, H, d_2 = d_2, x_0= c[0], y_0=c[1], M=M)
        A = A.tocsc()
        
        DOFs = spsolve(A,B)
        f = TrefftzFunction(V,DOFs)

        u_Trefft =  np.reshape([ f(x_, y_) for x_, y_ in zip( X.ravel(), Y.ravel()) ], [Ny,Nx])

        
        errors[i,j] = Domain.L2_norm(X,Y,u_exact-u_Trefft)/Domain.L2_norm(X,Y,u_exact)
# -

plot_hp_convergence(errors=errors,hs=hs, N_ths=N_ths, kappa_e=kappa_e, N_modes=N_modes, H=H, filename='fundamental_outside.pdf' )

np.savez(f"fundamental_outside{int(kappa_e)}.npz", errors=errors, hs=hs, N_ths = N_ths)

# # N_modes dependency

for k in [8, 16, 32]:
    print(np.floor(k*H/np.pi))

# +
K_s = [8, 16, 32 ]
len_range = 6
N_modes_ranges = [range(1,1+len_range), range(3,3+len_range), range(8,8+len_range)]
errors = np.zeros((3,len_range), dtype=np.float64 )

lambda_e = 2*np.pi/8
R = lambda_e
H = 1.


Ny = 100
Nx = int(R/H*Ny)
x = np.linspace(-R,R,Nx)
y = np.linspace(0,H,Ny)
X, Y = np.meshgrid(x,y)

M = 32
c = np.array([-R-0.5,H/3])

h = H/5
Domain = Waveguide(R=R,H=H) 
Domain.generate_mesh(h_max=h)




for i, k in enumerate(K_s):
    print(f"working on k: {k}")
    u_exact = np.reshape( GreenFunctionModes(np.real(k),Domain.H, np.stack([X.ravel(),Y.ravel()], axis=1), c[0], c[1], M=M), (Ny,Nx))
    for (j,N_modes) in enumerate(N_modes_ranges[i]):
        print(f"N_modes={N_modes}...")
        N_th = 15
        th_0 = np.e/np.pi # no correct direction in the basis
        V = TrefftzSpace(Domain.Omega, Nth, {"Omega_i" : k , "Omega_e" : k}, th_0 )
        # "UWVF" parameters
        a = 0.5
        b = 0.5
        d_1 = 0.5
        d_2 = 0.5

        A = AssembleMatrix(V, Domain.Edges, H, Np=N_modes, a=a, b=b, d_1=d_1, d_2=d_2)
        B = AssembleGreenRHS_left(V, Domain.Edges, k, H, d_2 = d_2, x_0= c[0], y_0=c[1], M=M)
        A = A.tocsc()
        
        DOFs = spsolve(A,B)
        f = TrefftzFunction(V,DOFs)

        u_Trefft =  np.reshape([ f(x_, y_) for x_, y_ in zip( X.ravel(), Y.ravel()) ], [Ny,Nx])

        
        errors[i,j] = Domain.L2_norm(X,Y,u_exact-u_Trefft)/Domain.L2_norm(X,Y,u_exact)

# -

for i,k in enumerate(K_s):
    plt.semilogy(N_modes_ranges[i],errors[i,:],'.-', label=f'$\\kappa = {k}$')
plt.ylabel('relative $L^2$ error')
plt.xlabel('Number of modes used for the approximation of the NtD map')
plt.legend()
plt.savefig('NtD_dependency.pdf')


# # Projection

# +
def integrate_ref(f,N=100):
    t = np.linspace(0,1,N+1)
    tx, ty = np.meshgrid(t,t)
    z = np.ma.array(f(tx,ty),mask=(tx+ty>1)) 
    # tx = np.ma.array( tx, )
    # ty = np.ma.array( ty, mask=(tx+ty>1))
    return z.sum()*(1/N*1/N)

integrate_ref(lambda x, y : x**2,2000)
#integrate_ref(lambda x, y : np.ones_like(x),2000)


# -

from integrators import fekete3 as fek3_int
# fek3_int(lambda x, y : np.ones_like(x), r_B=(1,0), r_C=(0,1))
# fek3_int(lambda x, y : x**2)


# # Test of the fek3 integrator

Domain = Waveguide(R=R,H=H)
Domain.generate_mesh(h_max=0.1*H) 
Omega = Domain.Omega
Domain.plot_mesh()

S = 0
for T in Omega.Elements():
    r_A = Omega.vertices[T.vertices[0].nr].point
    r_B = Omega.vertices[T.vertices[1].nr].point
    r_C = Omega.vertices[T.vertices[2].nr].point
    f = lambda x, y : 1
    S += fek3_int(f,r_A,r_B,r_C)
print(f'exact area: {2*R*H : .16f}, integrated: {S : .16f}, relative error: {np.abs(S-2*R*H)/(2*R*H) : .3e}')

Domain = Waveguide(R=R,H=H)
Domain.add_scatterer(scatterer_shape=ScattererShape.CIRCLE, scatterer_type=ScattererType.ABSORBING, params=((0,H/2),H/4))
Domain.generate_mesh(h_max=0.1*H) 
Omega = Domain.Omega
Domain.plot_mesh()

Domain.plot_scatterer_triangles()
# f = Omega.faces[0]
# els = list(Omega.Elements())
# els[f.elements[0].nr].mat

# ## Proyection of a mode
#
# at each triangle, the basis
#
# $$
# \left\{\varphi_n(\mathbf{x}) = e^{ik\mathbf{d}_n \cdot\mathbf{x}}\right\}_{n=1}^{N_p}
# $$
# is not orthogonal.
# $$
# u = \sum_{n=1}^{N_p} u_n \varphi_n
# $$
# so
# $$
# \sum_{n=1}^{N_p}\left\langle \varphi_n,\varphi_m\right\rangle u_n = \left\langle u_\mathrm{mode},\varphi_m \right\rangle\quad m=1,2,\dots, N_p
# $$

Domain = Waveguide(R=R,H=H)
Domain.generate_mesh(h_max=H) 
Omega = Domain.Omega
Domain.plot_mesh()


# +
r_A, r_B, r_C = [ Omega.vertices[v.nr].point for v in Omega.faces[0].vertices ] 

Nth = 7

t = 2
beta = np.emath.sqrt(kappa_e**2 - (np.pi*t/H)**2)



d_s = np.array( [[np.cos(th), np.sin(th)] for th in np.linspace(0,2*np.pi,Nth,endpoint=False)])
G = np.zeros( (Nth,Nth), dtype=np.complex128)
F = np.zeros(Nth,dtype=np.complex128)

for n in range(Nth):
    d_n = d_s[n]
    for m in range(Nth):
        d_m = d_s[m]
        G[m,n] = fek3_int( lambda x, y : np.exp(1j*kappa_e*((d_n - d_m)[0]*x + (d_n - d_m)[1]*y)), r_A=r_A, r_B=r_B, r_C=r_C)
        F[m] = fek3_int( lambda x, y : np.exp(1j*(beta - kappa_e*d_m)[0]*x)*np.exp(-1j*kappa_e*d_m[1]*y)*np.cos(t*np.pi*y/H), r_A=r_A, r_B=r_B, r_C=r_C) 

# +
coef_proy = np.linalg.solve(G,F)

x = np.linspace(-R,R,Nx)
y = np.linspace(-H,H,Ny)
X, Y = np.meshgrid(x,y)


u_proy = sum( [coef_proy[n]*np.exp(1j*kappa_e*(d_s[n][0]*X + d_s[n][1]*Y)) for n in range(Nth)])
# -

Domain.plot_field(X,Y,np.real(u_proy))

Omega = Domain.Omega

# ## condition number estimator: 

from scipy.sparse.linalg import svds
from scipy.sparse import sparray
def cond(A : sparray)-> np.float64:
    max_iter = 40000
    S_max = svds(A,k=1,which="LM", maxiter=max_iter, solver='lobpcg', return_singular_vectors=False)[0]
    S_min = svds(A,k=1,which="SM", maxiter=max_iter, solver='lobpcg',return_singular_vectors=False)[0]
    return S_max, S_min, S_max / S_min


# # Convergence



# +
refinements = range(3,10,1)
N_ths = [3,5,7,9,11,13,15]


N_ref = len(refinements)
N_N_th = len(N_ths)

hs = np.zeros((N_ref), dtype=np.float64)
conds = np.zeros((N_ref,N_N_th), dtype=np.float64)
errors = np.zeros((N_ref,N_N_th), dtype=np.float64)

factor = 8/8

kappa_e = factor*8.
lambda_e = 2*np.pi/kappa_e
R = factor*lambda_e
H = 1.


half_infinite = False
#c = (-0.6*H,0.6*H)
c = (0,0.6*H)

t=1

Ny = 100
Nx = int(R/H*Ny)
if half_infinite:
    x = np.linspace(0,2*R,Nx)
else:
    x = np.linspace(-R,R,Nx)
y = np.linspace(0,H,Ny)
X, Y = np.meshgrid(x,y)

M = 3200
modes = True
if modes:
    u_exact = np.reshape( GreenFunctionModes(kappa_e,Domain.H, np.stack([X.ravel(),Y.ravel()], axis=1), c[0], c[1], M=M), (Ny,Nx))
else:
    u_exact = np.reshape( GreenFunctionImages(kappa_e,Domain.H, np.stack([X.ravel(),Y.ravel()], axis=1), c[0], c[1], M=M), (Ny,Nx))


# t = 1
# u_exact = np.exp(1j*np.emath.sqrt(kappa_e**2 - (t*np.pi/H)**2)*X)*np.cos(t*np.pi*Y/H)
for (i,N) in enumerate(refinements):
    print(f"working on refinement: {N=}")
    h = H/(N)
    hs[i] = h
    #Domain = Waveguide(R=2*R,H=H, half_infinite=half_infinite) # CHANGE THIS
    Domain = Waveguide(R=R,H=H) 
    Domain.add_scatterer( ScattererShape.CIRCLE, ScattererType.SOUND_SOFT, (c, 0.1*H))
    #Domain.add_scatterer( ScattererShape.RECTANGLE, ScattererType.SOUND_SOFT, (c, 0.2*H, 0.2*H))
    # Domain.add_scatterer( ScattererShape.RECTANGLE, ScattererType.SOUND_SOFT, ((R,0.6*H), 0.2*H, 0.2*H))
    
    # Domain.add_fine_mesh_region(h_min=0.01*H)
    Domain.generate_mesh(h_max=h)

    side_ls = np.array([E.l for E in Domain.Edges])
    h_max = np.max(side_ls)
    print(f'{h_max=}')


    for (j,Nth) in enumerate(N_ths):
        # plt.close()
        print(f"N_theta={Nth}...")
        th_0 = np.e/np.pi # no correct direction in the basis
        V = TrefftzSpace(Domain.Omega, Nth, {"Omega_i" : kappa_e , "Omega_e" : kappa_e}, th_0 )
        N_modes = 15 #Number of modes for the DtN map
        # "UWVF" parameters
        a = 0.5
        b = 0.5
        d_1 = 0.5
        d_2 = 0.5

        # a = a*h_max/side_ls
        # b = b*h_max/side_ls
        # d_1 = d_1*h_max/side_ls
        # d_2 = d_2*h_max/side_ls


        A = AssembleMatrix(V, Domain.Edges, H, Np=N_modes, a=a, b=b, d_1=d_1, d_2=d_2)
        _, _, conds[i,j] = cond(A)
        #conds.append(np.linalg.cond(A.toarray()))
        B = AssembleGreenRHS(V, Domain.Edges, kappa_e, Domain.H, a=a, x_0=c[0], y_0=c[1], modes=modes, M=M)
        #B = AssembleRHS(V, Domain.Edges, kappa_e, H, d_2=d_2, t = t)

        A = A.tocsc()
        
        DOFs = spsolve(A,B)
        f = TrefftzFunction(V,DOFs)

        u_Trefft =  np.reshape([ f(x_, y_) for x_, y_ in zip( X.ravel(), Y.ravel()) ], [Ny,Nx])

        
        errors[i,j] = Domain.L2_norm(X,Y,u_exact-u_Trefft)/Domain.L2_norm(X,Y,u_exact)*100 
        # Domain.plot_field(X,Y,np.real(u_Trefft), show_edges=True)



    
    

# +
inches_per_dot = 1/72.27
cm2inch = 1/2.54 # inch per cm
columnwidth = 630.185 * inches_per_dot
columnwidth = 469.75502 * inches_per_dot
#columnwidth = 524.07272*inches_per_dot

columnwidth=columnwidth*0.9

left_margin = 1. * cm2inch # cm
right_margin = 1.*cm2inch  # cm
figure_width = columnwidth # cm
figure_height = columnwidth/1.4 # cm
top_margin = 1.*cm2inch    # cm
bottom_margin = 1.*cm2inch # cm

box_width = left_margin + figure_width + right_margin   # cm
box_height = top_margin + figure_height + bottom_margin # cm


# specifying the width and the height of the box in inches
# fig = plt.figure(figsize=(box_width,box_height))
# ax = fig.add_subplot(111)

fig, ax = plt.subplots(nrows=2,figsize=(box_width,box_height))

for err, h in zip(errors,hs):
    ax[0].semilogy(N_ths,err,'.-', label=f'$h_\\mathrm{{max}} = {h: .1e}$')



fig.suptitle(f'Fundamental solution at scatterer, $\\kappa={kappa_e}$, $H = {H:.0f}$, $R = \\lambda$, $M={N_modes}$ modes for de NtD map')

ax[0].set_xlabel('$N_p$')
ax[0].set_ylabel('$\\frac{\\left\\Vert u - u_h\\right\\Vert_2^2} {\\left\\Vert u \\right\\Vert_2^2}\\%$')
ax[0].yaxis.set_major_locator(mticker.LogLocator(numticks=999))
ax[0].yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
ax[0].set_xticks(range(3,17,2))
ax[0].grid(True,which="major",ls='--')
ax[0].legend()


for err, N_th in zip(errors.transpose(),N_ths):
    ax[1].semilogy(kappa_e*hs,err,'.-', label=f'$N_P = {N_th}$')

ax[1].set_xlabel('$\\kappa h$')
ax[1].set_ylabel('$\\frac{\\left\\Vert u - u_h\\right\\Vert_2^2} {\\left\\Vert u \\right\\Vert_2^2}\\%$')
ax[1].legend(loc="lower right")
ax[1].yaxis.set_major_locator(mticker.LogLocator(numticks=999))
ax[1].yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))

plt.grid(True,which="major",ls='--')




fig.subplots_adjust(left   = left_margin / box_width,
                    bottom = bottom_margin / box_height,
                    right  = 1. - right_margin / box_width,
                    top    = 1. - top_margin   / box_height)


#plt.text(x=11,y=1E-7,s='$\\propto 2^{-2.5p}$')
#plt.savefig('p-convergence_mode_1_reescaling.png')
plt.savefig(f'convergence_fundamental.pdf')


# +

for err, N_th in zip(errors.transpose(),N_ths):
    plt.semilogy(kappa_e*hs,err,'.-', label=f'$N_P = {N_th}$')

#plt.title(f'Propataging mode {t},\n $\\kappa={kappa_e}$, $H = {H:.0f}$, $R = \\lambda$, $M={N_modes}$ modes for de NtD')
#plt.title(f'Half waveguide,\n $\\kappa={kappa_e}$, $H = {H:.0f}$, $R = \\lambda$, $M={N_modes}$ modes for de NtD')
plt.title(f'Fundamental solution with {M} modes\n $\\kappa={kappa_e}$, $H = {H:.0f}$, $R = \\lambda$, $M={N_modes}$ modes for de NtD')

#plt.xlabel('$\\frac{1}{\\kappa h}$')
plt.xlabel('$\\kappa h$')
plt.ylabel('$\\frac{\\left\\Vert u - u_h\\right\\Vert_2^2} {\\left\\Vert u \\right\\Vert_2^2}\\%$')
plt.legend()
ax = plt.gca()
ax.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
ax.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))

plt.grid(True,which="major",ls='--')
plt.savefig(f'h-convergence_circle_k_{kappa_e:.3f}.png')

# +
for err, h in zip(errors,hs):
    plt.semilogy(N_ths,err,'.-', label=f'$h_\\mathrm{{max}} = {h: .1e}$')



#plt.title(f'Half waveguide,\n $\\kappa={kappa_e}$, $H = {H:.0f}$, $R = \\lambda$, $M={N_modes}$ modes for de NtD')
#plt.title(f'Propataging mode {t},\n $\\kappa={kappa_e}$, $H = {H:.0f}$, $R = \\lambda$, $M={N_modes}$ modes for de NtD')
plt.title(f'Fundamental solution with {M} modes\n $\\kappa={kappa_e}$, $H = {H:.0f}$, $R = \\lambda$, $M={N_modes}$ modes for de NtD')

plt.xlabel('$p$')
plt.ylabel('$\\frac{\\left\\Vert u - u_h\\right\\Vert_2^2} {\\left\\Vert u \\right\\Vert_2^2}\\%$')
ax = plt.gca()
ax.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
ax.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
plt.xticks(range(3,17,2))
plt.grid(True,which="major",ls='--')
plt.legend()
#plt.text(x=11,y=1E-7,s='$\\propto 2^{-2.5p}$')
#plt.savefig('p-convergence_mode_1_reescaling.png')
plt.savefig(f'p-convergence_circle_k_{kappa_e:.3f}.png')


# +
# conds = []
# refinements = range(3,10,1)
# N_ths = [3,5,7,9,11,13,15]


# N_ref = len(refinements)
# N_N_th = len(N_ths)

# hs = np.zeros((N_ref), dtype=np.float64)
# errors = np.zeros((N_ref,N_N_th), dtype=np.float64)

# factor = 8

# kappa_e = factor*8.
# lambda_e = 2*np.pi/kappa_e
# R = factor*lambda_e
# H = 1.


# half_infinite = False
# #c = (-0.6*H,0.6*H)
# c = (0,0.6*H)

# t=1

# Ny = 100
# Nx = int(R/H*Ny)
# if half_infinite:
#     x = np.linspace(0,2*R,Nx)
# else:
#     x = np.linspace(-R,R,Nx)
# y = np.linspace(0,H,Ny)
# X, Y = np.meshgrid(x,y)

# M = 3200
# modes = True
# if modes:
#     u_exact = np.reshape( GreenFunctionModes(kappa_e,Domain.H, np.stack([X.ravel(),Y.ravel()], axis=1), c[0], c[1], M=M), (Ny,Nx))
# else:
#     u_exact = np.reshape( GreenFunctionImages(kappa_e,Domain.H, np.stack([X.ravel(),Y.ravel()], axis=1), c[0], c[1], M=M), (Ny,Nx))


# # t = 1
# # u_exact = np.exp(1j*np.emath.sqrt(kappa_e**2 - (t*np.pi/H)**2)*X)*np.cos(t*np.pi*Y/H)
# for (i,N) in enumerate(refinements):
#     print(f"working on refinement: {N=}")
#     h = H/(N)
#     hs[i] = h
#     #Domain = Waveguide(R=2*R,H=H, half_infinite=half_infinite) # CHANGE THIS
#     Domain = Waveguide(R=R,H=H) 
#     Domain.add_scatterer( ScattererShape.CIRCLE, ScattererType.SOUND_SOFT, (c, 0.1*H))
#     #Domain.add_scatterer( ScattererShape.RECTANGLE, ScattererType.SOUND_SOFT, (c, 0.2*H, 0.2*H))
#     # Domain.add_scatterer( ScattererShape.RECTANGLE, ScattererType.SOUND_SOFT, ((R,0.6*H), 0.2*H, 0.2*H))
    
#     # Domain.add_fine_mesh_region(h_min=0.01*H)
#     Domain.generate_mesh(h_max=h)

#     side_ls = np.array([E.l for E in Domain.Edges])
#     h_max = np.max(side_ls)
#     print(f'{h_max=}')


#     for (j,Nth) in enumerate(N_ths):
#         # plt.close()
#         print(f"N_theta={Nth}...")
#         th_0 = np.e/np.pi # no correct direction in the basis
#         V = TrefftzSpace(Domain.Omega, Nth, {"Omega_i" : kappa_e , "Omega_e" : kappa_e}, th_0 )
#         N_modes = 15 #Number of modes for the DtN map
#         # "UWVF" parameters
#         a = 0.5
#         b = 0.5
#         d_1 = 0.5
#         d_2 = 0.5



#         A = AssembleMatrix(V, Domain.Edges, H, Np=N_modes, a=a, b=b, d_1=d_1, d_2=d_2)
#         #conds.append(np.linalg.cond(A.toarray()))
#         B = AssembleGreenRHS(V, Domain.Edges, kappa_e, Domain.H, a=a, x_0=c[0], y_0=c[1], modes=modes, M=M)
#         #B = AssembleRHS(V, Domain.Edges, kappa_e, H, d_2=d_2, t = t)

#         A = A.tocsc()
#         DOFs = spsolve(A,B)
#         f = TrefftzFunction(V,DOFs)

#         u_Trefft =  np.reshape([ f(x_, y_) for x_, y_ in zip( X.ravel(), Y.ravel()) ], [Ny,Nx])

        
#         errors[i,j] = Domain.L2_norm(X,Y,u_exact-u_Trefft)/Domain.L2_norm(X,Y,u_exact)*100 
#         # Domain.plot_field(X,Y,np.real(u_Trefft), show_edges=True)



# +
refinements = range(3,10,1)
N_ths = [3,5,7,9,11,13,15]


N_ref = len(refinements)
N_N_th = len(N_ths)

hs = np.zeros((N_ref), dtype=np.float64)
conds = np.zeros((N_ref,N_N_th), dtype=np.float64)
errors = np.zeros((N_ref,N_N_th), dtype=np.float64)

factor = 8/8

kappa_e = factor*8.
lambda_e = 2*np.pi/kappa_e
R = factor*lambda_e
H = 1.


c = (0,0.6*H)

t=1

Ny = 100
Nx = int(R/H*Ny)
x = np.linspace(-R,R,Nx)
y = np.linspace(0,H,Ny)
X, Y = np.meshgrid(x,y)



t = 1
u_exact = np.exp(1j*np.emath.sqrt(kappa_e**2 - (t*np.pi/H)**2)*X)*np.cos(t*np.pi*Y/H)

for (i,N) in enumerate(refinements):
    print(f"working on refinement: {N=}")
    h = H/(N)
    hs[i] = h
    Domain = Waveguide(R=R,H=H) 
    Domain.generate_mesh(h_max=h)

    side_ls = np.array([E.l for E in Domain.Edges])
    h_max = np.max(side_ls)
    print(f'{h_max=}')


    for (j,Nth) in enumerate(N_ths):
        print(f"N_theta={Nth}...")
        th_0 = np.e/np.pi # no correct direction in the basis
        V = TrefftzSpace(Domain.Omega, Nth, {"Omega_i" : kappa_e , "Omega_e" : kappa_e}, th_0 )
        N_modes = 15 #Number of modes for the DtN map
        # "UWVF" parameters
        a = 0.5
        b = 0.5
        d_1 = 0.5
        d_2 = 0.5



        A = AssembleMatrix(V, Domain.Edges, H, Np=N_modes, a=a, b=b, d_1=d_1, d_2=d_2)
        _,_, conds[i,j] = cond(A)
        B = AssembleRHS(V, Domain.Edges, kappa_e, H, d_2=d_2, t = t)

        A = A.tocsc()
        DOFs = spsolve(A,B)
        f = TrefftzFunction(V,DOFs)

        u_Trefft =  np.reshape([ f(x_, y_) for x_, y_ in zip( X.ravel(), Y.ravel()) ], [Ny,Nx])

        
        errors[i,j] = Domain.L2_norm(X,Y,u_exact-u_Trefft)/Domain.L2_norm(X,Y,u_exact)*100 
        


    
    

# +
inches_per_dot = 1/72.27
cm2inch = 1/2.54 # inch per cm
columnwidth = 630.185 * inches_per_dot
columnwidth = 469.75502 * inches_per_dot
#columnwidth = 524.07272*inches_per_dot

columnwidth=columnwidth*0.9

left_margin = 1. * cm2inch # cm
right_margin = 1.*cm2inch  # cm
figure_width = columnwidth # cm
figure_height = columnwidth/1.4 # cm
top_margin = 1.*cm2inch    # cm
bottom_margin = 1.*cm2inch # cm

box_width = left_margin + figure_width + right_margin   # cm
box_height = top_margin + figure_height + bottom_margin # cm


# specifying the width and the height of the box in inches
# fig = plt.figure(figsize=(box_width,box_height))
# ax = fig.add_subplot(111)

fig, ax = plt.subplots(nrows=2,figsize=(box_width,box_height))

for err, h in zip(errors,hs):
    ax[0].semilogy(N_ths,err,'.-', label=f'$h_\\mathrm{{max}} = {h: .1e}$')



fig.suptitle(f'Propataging mode {t}, $\\kappa={kappa_e}$, $H = {H:.0f}$, $R = \\lambda$, $M={N_modes}$ modes for de NtD map')

ax[0].set_xlabel('$N_p$')
ax[0].set_ylabel('$\\frac{\\left\\Vert u - u_h\\right\\Vert_2^2} {\\left\\Vert u \\right\\Vert_2^2}\\%$')
ax[0].yaxis.set_major_locator(mticker.LogLocator(numticks=999))
ax[0].yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
ax[0].set_xticks(range(3,17,2))
ax[0].grid(True,which="major",ls='--')
ax[0].legend()


for err, N_th in zip(errors.transpose(),N_ths):
    ax[1].semilogy(kappa_e*hs,err,'.-', label=f'$N_P = {N_th}$')

ax[1].set_xlabel('$\\kappa h$')
ax[1].set_ylabel('$\\frac{\\left\\Vert u - u_h\\right\\Vert_2^2} {\\left\\Vert u \\right\\Vert_2^2}\\%$')
ax[1].legend(loc="lower right")
ax[1].yaxis.set_major_locator(mticker.LogLocator(numticks=999))
ax[1].yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))

plt.grid(True,which="major",ls='--')




fig.subplots_adjust(left   = left_margin / box_width,
                    bottom = bottom_margin / box_height,
                    right  = 1. - right_margin / box_width,
                    top    = 1. - top_margin   / box_height)


#plt.text(x=11,y=1E-7,s='$\\propto 2^{-2.5p}$')
#plt.savefig('p-convergence_mode_1_reescaling.png')
plt.savefig(f'convergence.pdf')

# -

plt.semilogy(conds[0,:])
plt.semilogy(conds[1,:])
plt.semilogy(conds[2,:])

conds.shape


