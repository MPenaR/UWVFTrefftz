# Waveguide with a scatterer inside
from netgen.geom2d import SplineGeometry
from ngsolve import Mesh, H1, pml, BilinearForm, SymbolicBFI, SymbolicLFI, grad, CoefficientFunction
from ngsolve import exp, cos, sqrt, GridFunction, BND, LinearForm, x, y
import numpy as np


def u_FEM_SOUNDSOFT(R = 10., H=2., rad = 0.2, c = (0.,1.), n=0, k=8., X=None, Y=None):
    porder=5  # polynomial FEM order
    delta_PML = 2 * 2*np.pi/k
    geo = SplineGeometry()
    hmaxf = 2*np.pi/k / 10
    p1,p2,p3,p4 = [ geo.AppendPoint(x,y) for x,y in [ (-R,0), (R,0),(R,H),(-R,H)]]
    geo.Append (["line", p1, p2],leftdomain=1,rightdomain=0,bc="pipe")
    geo.Append (["line", p2, p3],leftdomain=1,rightdomain=2)
    geo.Append (["line", p3, p4],leftdomain=1,rightdomain=0,bc="pipe")
    geo.Append (["line", p4, p1],leftdomain=1,rightdomain=3)
    geo.AddCircle(c, rad, leftdomain=0, rightdomain=1,bc="dirichlet")
    p5,p6 = [ geo.AppendPoint(x,y) for x,y in [ (-R-delta_PML,0),
                                                (-R-delta_PML,H)]]
    
    geo.Append (["line", p5, p1],leftdomain=3,rightdomain=0,bc="pipe")
    geo.Append (["line", p6, p5],leftdomain=3,rightdomain=0,bc="PML")
    geo.Append (["line", p4, p6],leftdomain=3,rightdomain=0,bc="pipe")
    p7,p8 = [ geo.AppendPoint(x,y) for x,y in [ (R+delta_PML,0),
                                                (R+delta_PML,H)]]
    geo.Append (["line", p2, p7],leftdomain=2,rightdomain=0,bc="pipe")
    geo.Append (["line", p7, p8],leftdomain=2,rightdomain=0,bc="PML")
    geo.Append (["line", p8, p3],leftdomain=2,rightdomain=0,bc="pipe")
    geo.SetMaterial(1, "fluid")
    geo.SetMaterial(2, "PML_right")
    geo.SetMaterial(3, "PML_left") 
    geo.SetDomainMaxH(3,hmaxf)
    geo.SetDomainMaxH(2,hmaxf)
    geo.SetDomainMaxH(1,hmaxf)
    ngmesh=geo.GenerateMesh()
    mesh = Mesh(ngmesh)

    mesh.Curve(porder)

    fes = H1(mesh, order=porder, complex=True, dirichlet="dirichlet|PML")

    factor = 0.1

    mesh.SetPML(pml.HalfSpace(point=(R,H/2),normal=(1,0),alpha=factor*(4+2*1j)),
                    "PML_right")
    mesh.SetPML(pml.HalfSpace(point=(-R,H/2),normal=(-1,0),alpha=factor*(4+2*1j)),
                    "PML_left")

    u  = fes.TrialFunction()
    v  = fes.TestFunction()

    a=BilinearForm(fes)
    a += SymbolicBFI(grad(u)*grad(v) - k**2*u*v)

    a.Assemble()
    Ainv=a.mat.Inverse(freedofs=fes.FreeDofs())



    u_inc = CoefficientFunction(0.+0.*1j)
    beta_n = np.sqrt(complex(k**2 - (n*np.pi/H)**2))
    u_inc += exp( 1j*beta_n*x)*cos(n*np.pi*y/H)
    u_sc = GridFunction(fes)
    u_sc.vec[:]=0+0*1J
    u_sc.Set(-u_inc, BND, definedon=mesh.Boundaries("dirichlet"))

    f = LinearForm(fes)
    f.Assemble()
    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * u_sc.vec
    u_sc.vec.data += Ainv * res
    u_tot = u_sc + u_inc

    print(f'NDOF: {fes.ndof}')

    if X is not None:
        Ny, Nx = X.shape
        x_vec = X[0,:]
        y_vec = Y[:,0]
        
        U_tot = np.full_like(X,fill_value=np.nan, dtype=np.complex128)

        for i in range(Ny):
            for j in range(Nx):
                x_ = x_vec[j]
                y_ = y_vec[i]
                if (x_ - c[0])**2 + (y_ - c[1])**2 <= rad**2:
                    pass
                else:
                    U_tot[i,j] = u_tot(mesh(x_,y_))
        return U_tot
    else:
        return u_tot, mesh

def u_FEM_PENETRABLE(R = 10., H=2., rad = 0.2, c = (0.,1.), n=0, k_e=8., k_i = 12.,  X=None, Y=None):

    # from netgen.geom2d import SplineGeometry
    # from ngsolve import Mesh, H1, pml, BilinearForm, SymbolicBFI, grad, CoefficientFunction
    # from ngsolve import exp, cos, sqrt, GridFunction, BND, LinearForm, x, y, SymbolicLFI
    # import numpy as np


    porder=5  # polynomial FEM order
    delta_PML = 4 * 2*np.pi/k_e
    geo = SplineGeometry()
    hmax_e = 2*np.pi/k_e / 10
    hmax_i = 2*np.pi/np.real(k_i) / 10
    p1,p2,p3,p4 = [ geo.AppendPoint(x,y) for x,y in [ (-R,0), (R,0),(R,H),(-R,H)]]
    geo.Append (["line", p1, p2],leftdomain=1,rightdomain=0,bc="pipe")
    geo.Append (["line", p2, p3],leftdomain=1,rightdomain=2)
    geo.Append (["line", p3, p4],leftdomain=1,rightdomain=0,bc="pipe")
    geo.Append (["line", p4, p1],leftdomain=1,rightdomain=3)
    geo.AddCircle(c, rad, leftdomain=4, rightdomain=1)
    p5,p6 = [ geo.AppendPoint(x,y) for x,y in [ (-R-delta_PML,0),
                                                (-R-delta_PML,H)]]
    
    geo.Append (["line", p5, p1],leftdomain=3,rightdomain=0,bc="pipe")
    geo.Append (["line", p6, p5],leftdomain=3,rightdomain=0,bc="PML")
    geo.Append (["line", p4, p6],leftdomain=3,rightdomain=0,bc="pipe")
    p7,p8 = [ geo.AppendPoint(x,y) for x,y in [ (R+delta_PML,0),
                                                (R+delta_PML,H)]]
    geo.Append (["line", p2, p7],leftdomain=2,rightdomain=0,bc="pipe")
    geo.Append (["line", p7, p8],leftdomain=2,rightdomain=0,bc="PML")
    geo.Append (["line", p8, p3],leftdomain=2,rightdomain=0,bc="pipe")
    geo.SetMaterial(1, "fluid")
    geo.SetMaterial(2, "PML_right")
    geo.SetMaterial(3, "PML_left")
    geo.SetMaterial(4, "scatterer")
    geo.SetDomainMaxH(4,hmax_i) 
    geo.SetDomainMaxH(3,hmax_e)
    geo.SetDomainMaxH(2,hmax_e)
    geo.SetDomainMaxH(1,hmax_e)
    ngmesh=geo.GenerateMesh()
    mesh = Mesh(ngmesh)

    mesh.Curve(porder)

    fes = H1(mesh, order=porder, complex=True, dirichlet="PML")

    factor = 0.1

    mesh.SetPML(pml.HalfSpace(point=(R,H),normal=(1,0),alpha=factor*(4+2*1j)),
                    "PML_right")
    mesh.SetPML(pml.HalfSpace(point=(-R,H),normal=(-1,0),alpha=factor*(4+2*1j)),
                    "PML_left")

    u  = fes.TrialFunction()
    v  = fes.TestFunction()


    k=CoefficientFunction([k_i if mat=="scatterer" else k_e for mat in mesh.GetMaterials()])


    a=BilinearForm(fes)
    a += SymbolicBFI(grad(u)*grad(v) - k**2*u*v)

    a.Assemble()
    Ainv=a.mat.Inverse(freedofs=fes.FreeDofs())



    u_inc = CoefficientFunction(0.+0.*1j)
    beta_n = np.sqrt(complex(k_e**2 - (n*np.pi/H)**2))
    u_inc += exp( 1j*beta_n*x)*cos(n*np.pi*y/H)
    u_sc = GridFunction(fes)

    f = LinearForm(fes)
    f += SymbolicLFI(k_e**2*( (k_i/k_e)**2 - 1)*u_inc*v, definedon=mesh.Materials('scatterer'))
    f.Assemble()
    

    u_sc.vec.data += Ainv * f.vec
    u_tot = u_sc + u_inc

    if X is not None:
        Ny, Nx = X.shape
        x_vec = X[0,:]
        y_vec = Y[:,0]
        
        U_tot = np.zeros_like(X, dtype=np.complex128)

        for i in range(Ny):
            for j in range(Nx):
                x_ = x_vec[j]
                y_ = y_vec[i]
                U_tot[i,j] = u_tot(mesh(x_,y_))
        return U_tot
    else:
        return u_tot, mesh


if __name__=='__main__':
    from ngsolve import VTKOutput
    R = 10
    H = 1
    rad = 0.2


    u_tot, mesh = u_FEM_SOUNDSOFT(R=R, H=H, rad=rad, n=2)


    vtk = VTKOutput(ma=mesh, coefs=[u_tot.real, sqrt(u_tot.real**2 + u_tot.imag**2)], names=['Re(u_tot)', '|u_tot|'])

    vtk.Do()