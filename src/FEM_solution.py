# Waveguide with a scatterer inside
from domains import ScattererShape, ScattererType
from netgen.geom2d import SplineGeometry
from ngsolve import Mesh, H1, pml, BilinearForm, SymbolicBFI, SymbolicLFI, grad, CoefficientFunction
from ngsolve import exp, cos, sqrt, GridFunction, BND, LinearForm, x, y
import numpy as np
import numpy.typing as npt


real_array = npt.NDArray[np.float64]

def FEM_solution(R : np.float64, H : np.float64, 
                 params : dict[str,np.float64  | real_array],
                 scatterer_shape : ScattererShape,
                 scatterer_type : ScattererType,
                 n : np.int64,
                 k_e : np.float64,
                 k_i : np.float64 | np.complex128, 
                 polynomial_order = 5,
                 X=real_array, Y=real_array):
    """Uses a PML and a very fine mesh to solve the problem by the finite element method."""

    delta_PML = 3 * 2*np.pi/k_e
    
    hmax_e = 2*np.pi/k_e / 10
    if scatterer_type == ScattererType.PENETRABLE:
        hmax_i = 2*np.pi/np.real(k_i) / 10


# perhaps clean this
    geo = SplineGeometry()
    p1, p2, p3, p4 = [ geo.AppendPoint(x,y) for x,y in [ (-R,0), (R,0), (R,H), (-R,H) ]]
    geo.Append (["line", p1, p2], leftdomain=1, rightdomain=0, bc="wall")
    geo.Append (["line", p2, p3], leftdomain=1, rightdomain=2)
    geo.Append (["line", p3, p4], leftdomain=1, rightdomain=0, bc="wall")
    geo.Append (["line", p4, p1], leftdomain=1, rightdomain=3)

    match scatterer_shape:
        case ScattererShape.CIRCLE:
            rad = params["rad"]
            c = params["c"]
            match scatterer_type:
                case ScattererType.PENETRABLE:
                    geo.AddCircle(c, rad, leftdomain=4, rightdomain=1)
                case ScattererType.SOUND_HARD | ScattererType.SOUND_SOFT:
                    geo.AddCircle(c, rad, leftdomain=0, rightdomain=1,bc="dirichlet")

        case ScattererShape.RECTANGLE:
            height = params["height"]
            width = params["width"]
            c = params["c"]
            match scatterer_type:
                case ScattererType.PENETRABLE:
                    geo.AddRectangle(p1=(c[0]-width/2,c[1]-height/2), p2=(c[0]+width/2,c[1]+height/2), leftdomain=4, rightdomain=1)    
                case ScattererType.SOUND_HARD | ScattererType.SOUND_SOFT:
                    geo.AddRectangle(p1=(c[0]-width/2,c[1]-height/2), p2=(c[0]+width/2,c[1]+height/2), leftdomain=0, rightdomain=1, bc="dirichlet")

    p5, p6 = [ geo.AppendPoint(x,y) for x,y in [ (-R-delta_PML,0),
                                                (-R-delta_PML,H)]]
    
    geo.Append (["line", p5, p1], leftdomain=3, rightdomain=0, bc="wall")
    geo.Append (["line", p6, p5], leftdomain=3, rightdomain=0, bc="PML")
    geo.Append (["line", p4, p6], leftdomain=3, rightdomain=0, bc="pwall")

    p7, p8 = [ geo.AppendPoint(x,y) for x,y in [ (R+delta_PML,0),
                                                (R+delta_PML,H)]]
    
    geo.Append (["line", p2, p7],leftdomain=2,rightdomain=0,bc="wall")
    geo.Append (["line", p7, p8],leftdomain=2,rightdomain=0,bc="PML")
    geo.Append (["line", p8, p3],leftdomain=2,rightdomain=0,bc="wall")
    geo.SetMaterial(1, "fluid")
    geo.SetMaterial(2, "PML_right")
    geo.SetMaterial(3, "PML_left")
    match scatterer_type:
        case ScattererType.PENETRABLE:
                geo.SetMaterial(4, "scatterer")
                geo.SetDomainMaxH(4, hmax_i) 
    geo.SetDomainMaxH(3,hmax_e)
    geo.SetDomainMaxH(2,hmax_e)
    geo.SetDomainMaxH(1,hmax_e)
    ngmesh=geo.GenerateMesh()
    mesh = Mesh(ngmesh)

    mesh.Curve(polynomial_order)

    match scatterer_type:
        case ScattererType.PENETRABLE:
            V = H1(mesh, order=polynomial_order, complex=True, dirichlet="PML")
        case ScattererType.SOUND_HARD | ScattererType.SOUND_SOFT:
            V = H1(mesh, order=polynomial_order, complex=True, dirichlet="dirichlet|PML")

    factor = 0.5

    mesh.SetPML( pml.HalfSpace(point=(R,H/2),normal=(1,0),alpha=factor*(4+2*1j)),
                    "PML_right")
    mesh.SetPML( pml.HalfSpace(point=(-R,H/2),normal=(-1,0),alpha=factor*(4+2*1j)),
                    "PML_left")


    u  = V.TrialFunction()
    v  = V.TestFunction()

    match scatterer_type:
        case ScattererType.SOUND_SOFT:
            k = k_e
        case ScattererType.SOUND_HARD:
            k = k_e
        case ScattererType.PENETRABLE:
            k = CoefficientFunction([k_i if mat=="scatterer" else k_e for mat in mesh.GetMaterials()])
    

    a = BilinearForm(V)
    a += SymbolicBFI(grad(u)*grad(v) - k**2*u*v)
    a.Assemble()
    Ainv = a.mat.Inverse(freedofs=V.FreeDofs())

    match scatterer_type:
        case ScattererType.SOUND_SOFT:
            u_inc = CoefficientFunction(0.+0.*1j)
            beta_n = np.emath.sqrt(complex(k**2 - (n*np.pi/H)**2))
            u_inc += exp( 1j*beta_n*x)*cos(n*np.pi*y/H)
            u_sc = GridFunction(V)
            u_sc.vec[:]=0+0*1J
            u_sc.Set(-u_inc, BND, definedon=mesh.Boundaries("dirichlet"))

            f = LinearForm(V)
            f.Assemble()
            res = f.vec.CreateVector()
            res.data = f.vec - a.mat * u_sc.vec
            u_sc.vec.data += Ainv * res

        case ScattererType.SOUND_HARD:
            pass
        case ScattererType.PENETRABLE:
            u_inc = CoefficientFunction(0.+0.*1j)
            beta_n = np.emath.sqrt(complex(k_e**2 - (n*np.pi/H)**2))
            u_inc += exp( 1j*beta_n*x)*cos(n*np.pi*y/H)
            u_sc = GridFunction(V)


            f = LinearForm(V)
            # f += SymbolicLFI(k_e**2*( (k_i/k_e)**2 - 1)*u_inc*v, definedon=mesh.Materials('scatterer'))

            f.Assemble()
            u_sc.vec.data += Ainv * f.vec

    u_tot = u_sc + u_inc
    print(f'NDOF: {V.ndof}')

    Ny, Nx = X.shape
    x_vec = X[0,:]
    y_vec = Y[:,0]
    

    match scatterer_type:
        case ScattererType.PENETRABLE:
            U_tot = np.zeros_like(X, dtype=np.complex128)

            for i in range(Ny):
                for j in range(Nx):
                    x_ = x_vec[j]
                    y_ = y_vec[i]
                    U_tot[i,j] = u_tot(mesh(x_,y_))
        case ScattererType.SOUND_HARD | ScattererType.SOUND_SOFT:
            U_tot = np.full_like(X,fill_value=np.nan, dtype=np.complex128)
            match scatterer_shape:
                case ScattererShape.CIRCLE:
                    for i in range(Ny):
                        for j in range(Nx):
                            x_ = x_vec[j]
                            y_ = y_vec[i]
                            if (x_ - c[0])**2 + (y_ - c[1])**2 <= rad**2:
                                pass
                            else:
                                U_tot[i,j] = u_tot(mesh(x_,y_))

                case ScattererShape.RECTANGLE:
                    for i in range(Ny):
                        for j in range(Nx):
                            x_ = x_vec[j]
                            y_ = y_vec[i]
                            if np.abs(x_ - c[0]) <= width/2 and np.abs(y_ - c[1]) <= height/2:
                                pass
                            else:
                                U_tot[i,j] = u_tot(mesh(x_,y_))
        
    return U_tot


if __name__=='__main__':
    from ngsolve import VTKOutput
    R = 10
    H = 1
    rad = 0.2

    Ny = 10
    Nx = int(R/H)*Ny


    x = np.linspace(-R,R,Nx)
    y = np.linspace(0,R,Ny)
    X, Y = np.meshgrid(x,y)


    u_tot, mesh = FEM_solution(R=R, H=H, scatterer_shape=ScattererShape.CIRCLE, scatterer_type=ScattererType.SOUND_SOFT, n=2,
                               k_e=8, k_i=8,X=X, Y = Y )


    vtk = VTKOutput(ma=mesh, coefs=[u_tot.real, sqrt(u_tot.real**2 + u_tot.imag**2)], names=['Re(u_tot)', '|u_tot|'])

    vtk.Do()