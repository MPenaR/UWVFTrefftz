# Waveguide with a scatterer inside
import scipy.io as sio
from netgen.geom2d import *
from ngsolve import *
#import netgen.gui
import numpy as np


Dirichlet=True  # set to False for the penetrable case
kappa=14. # wavenumber

y1=-2.   # X coordinate of the source line
Nres= 5 #2*round(1./(2*np.pi/kappa)*10)  # number of output points
y2,rstep=np.linspace(1/(Nres+1),1,Nres,endpoint=False,retstep=True)

collectL=-3. #-abs(y1) # X coordinate of the measuring line (left side)
collectR=3. #abs(y1)    # X coordinate of the measuring line (right side)

H=1.     # height of the waveguide
epsm=2.  # n inside the scatterer

Rsc1=.07    #1st scatterer's radius
y1sc1 =0.0 #1st scatterer's X coordinate of the center
y2sc1 =0.3 #1st scatterer's Y coordinate of the center

Rsc2 =.07    #2nd scatterer's radius
y1sc2 =0.3 #2nd scatterer's X coordinate of the center
y2sc2 =0.5 #2nd scatterer's Y coordinate of the center

Nterm=20  # number of terms in the expansion of the Greens function
porder=5  # polynomial FEM order

def WG_with_circle(hmaxf=0.1,hmaxs=0.1,pml_dist_plus=2,
                       pml_dist_minus=2,delta_pml=1,H=1,R_scat=Rsc1,
                       circ_O=(y1sc1,y2sc1),collL=-1,collR=1):
    geo = SplineGeometry()
    p1,p2,p3,p4 = [ geo.AppendPoint(x,y) for x,y in [ (-pml_dist_minus,0),
                        (pml_dist_plus,0),(pml_dist_plus,H),(-pml_dist_minus,H)]]
    p2a=geo.AppendPoint(collL,0)
    p3a=geo.AppendPoint(collL,H)
    p2b=geo.AppendPoint(collR,0)
    p3b=geo.AppendPoint(collR,H)
    geo.Append (["line", p1, p2a],leftdomain=5,rightdomain=0,bc="pipe")
    geo.Append (["line", p2a, p2b],leftdomain=1,rightdomain=0,bc="pipe")
    geo.Append (["line", p2b, p2],leftdomain=6,rightdomain=0,bc="pipe")
    geo.Append (["line", p2, p3],leftdomain=6,rightdomain=2)
    geo.Append (["line", p3, p3b],leftdomain=6,rightdomain=0,bc="pipe")
    geo.Append (["line", p3b, p3a],leftdomain=1,rightdomain=0,bc="pipe")
    geo.Append (["line", p3a, p4],leftdomain=5,rightdomain=0,bc="pipe")
    geo.Append (["line", p4, p1],leftdomain=5,rightdomain=3)
    geo.Append (["line", p2a, p3a],leftdomain=5,rightdomain=1,bc="collectL")
    geo.Append (["line", p2b, p3b],leftdomain=1,rightdomain=6,bc="collectR")
    if Dirichlet:
        geo.AddCircle(circ_O, R_scat, leftdomain=0, rightdomain=1,bc="dirichlet")
    else:
        geo.AddCircle(circ_O, R_scat, leftdomain=4, rightdomain=1)
    p5,p6 = [ geo.AppendPoint(x,y) for x,y in [ (-pml_dist_minus-delta_pml,0),
                                                (-pml_dist_minus-delta_pml,H)]]
    p2b=geo.AppendPoint(collR,0)
    p3b=geo.AppendPoint(collR,H)
    
    geo.Append (["line", p5, p1],leftdomain=3,rightdomain=0,bc="pipe")
    geo.Append (["line", p6, p5],leftdomain=3,rightdomain=0,bc="PML")
    geo.Append (["line", p4, p6],leftdomain=3,rightdomain=0,bc="pipe")
    p7,p8 = [ geo.AppendPoint(x,y) for x,y in [ (pml_dist_plus+delta_pml,0),
                                                (pml_dist_plus+delta_pml,H)]]
    geo.Append (["line", p2, p7],leftdomain=2,rightdomain=0,bc="pipe")
    geo.Append (["line", p7, p8],leftdomain=2,rightdomain=0,bc="PML")
    geo.Append (["line", p8, p3],leftdomain=2,rightdomain=0,bc="pipe")
    geo.SetMaterial(1, "fluid")
    geo.SetMaterial(2, "PML_right")
    geo.SetMaterial(3, "PML_left") 
    geo.SetMaterial(4, "scatterer") 
    geo.SetMaterial(5, "fluid") 
    geo.SetMaterial(6, "fluid") 
    geo.SetDomainMaxH(4,hmaxs)
    geo.SetDomainMaxH(3,hmaxf)
    geo.SetDomainMaxH(2,hmaxf)
    geo.SetDomainMaxH(1,hmaxf)
    geo.SetDomainMaxH(5,hmaxf)
    geo.SetDomainMaxH(6,hmaxf)
    ngmesh=geo.GenerateMesh()
    mesh = Mesh(ngmesh)
    return mesh

def WG_with_circles(hmaxf=0.1,hmaxs=0.1,pml_dist_plus=2,
                       pml_dist_minus=2,delta_pml=1,H=1,
                       R_scat0=Rsc1,circ_O=(y1sc1,y2sc1),
                       R_scat1=Rsc2,circ_1=(y1sc2,y2sc2),
                       collL=-1,collR=1):
    geo = SplineGeometry()
    p1,p2,p3,p4 = [ geo.AppendPoint(x,y) for x,y in [ (-pml_dist_minus,0),
                        (pml_dist_plus,0),(pml_dist_plus,H),(-pml_dist_minus,H)]]
    p2a=geo.AppendPoint(collL,0)
    p3a=geo.AppendPoint(collL,H)
    p2b=geo.AppendPoint(collR,0)
    p3b=geo.AppendPoint(collR,H)
    geo.Append (["line", p1, p2a],leftdomain=5,rightdomain=0,bc="pipe")
    geo.Append (["line", p2a, p2b],leftdomain=1,rightdomain=0,bc="pipe")
    geo.Append (["line", p2b, p2],leftdomain=6,rightdomain=0,bc="pipe")
    geo.Append (["line", p2, p3],leftdomain=6,rightdomain=2)
    geo.Append (["line", p3, p3b],leftdomain=6,rightdomain=0,bc="pipe")
    geo.Append (["line", p3b, p3a],leftdomain=1,rightdomain=0,bc="pipe")
    geo.Append (["line", p3a, p4],leftdomain=5,rightdomain=0,bc="pipe")
    geo.Append (["line", p4, p1],leftdomain=5,rightdomain=3)
    geo.Append (["line", p2a, p3a],leftdomain=5,rightdomain=1,bc="collectL")
    geo.Append (["line", p2b, p3b],leftdomain=1,rightdomain=6,bc="collectR")
    if Dirichlet:
        geo.AddCircle(circ_O, R_scat0, leftdomain=0, rightdomain=1,bc="dirichlet")
        geo.AddCircle(circ_1, R_scat1, leftdomain=0, rightdomain=1,bc="dirichlet")
    else:
        geo.AddCircle(circ_O, R_scat0, leftdomain=4, rightdomain=1)
        geo.AddCircle(circ_1, R_scat1, leftdomain=4, rightdomain=1)
    p5,p6 = [ geo.AppendPoint(x,y) for x,y in [ (-pml_dist_minus-delta_pml,0),
                                                (-pml_dist_minus-delta_pml,H)]]
    p2b=geo.AppendPoint(collR,0)
    p3b=geo.AppendPoint(collR,H)
    
    geo.Append (["line", p5, p1],leftdomain=3,rightdomain=0,bc="pipe")
    geo.Append (["line", p6, p5],leftdomain=3,rightdomain=0,bc="PML")
    geo.Append (["line", p4, p6],leftdomain=3,rightdomain=0,bc="pipe")
    p7,p8 = [ geo.AppendPoint(x,y) for x,y in [ (pml_dist_plus+delta_pml,0),
                                                (pml_dist_plus+delta_pml,H)]]
    geo.Append (["line", p2, p7],leftdomain=2,rightdomain=0,bc="pipe")
    geo.Append (["line", p7, p8],leftdomain=2,rightdomain=0,bc="PML")
    geo.Append (["line", p8, p3],leftdomain=2,rightdomain=0,bc="pipe")
    geo.SetMaterial(1, "fluid")
    geo.SetMaterial(2, "PML_right")
    geo.SetMaterial(3, "PML_left") 
    geo.SetMaterial(4, "scatterer") 
    geo.SetMaterial(5, "fluid") 
    geo.SetMaterial(6, "fluid") 
    geo.SetDomainMaxH(4,hmaxs)
    geo.SetDomainMaxH(3,hmaxf)
    geo.SetDomainMaxH(2,hmaxf)
    geo.SetDomainMaxH(1,hmaxf)
    geo.SetDomainMaxH(5,hmaxf)
    geo.SetDomainMaxH(6,hmaxf)
    ngmesh=geo.GenerateMesh()
    mesh = Mesh(ngmesh)
    return mesh

lam_f=2*np.pi/kappa
lam_s=2*np.pi/kappa/np.sqrt(epsm)
hmax_f=lam_f/8
hmax_s=lam_s/8
pdp=collectR+1+2*lam_f
pdm=-collectL+1+2*lam_f # collectL is negative
delta=2*lam_f

##in case the scatterer is a circle alone:
mesh=WG_with_circle(hmaxf=hmax_f,pml_dist_minus=pdm,
                                 hmaxs=hmax_s,pml_dist_plus=pdp,
                                 delta_pml=delta,H=H,collL=collectL,
                                 collR=collectR)

#in case the scatterer is two circles together:
#mesh=WG_with_circles(hmaxf=hmax_f,pml_dist_minus=pdm,
#                                 hmaxs=hmax_s,pml_dist_plus=pdp,
#                                 delta_pml=delta,H=H,collL=collectL,
#                                 collR=collectR)


mesh.Curve(porder)

nv=specialcf.normal(mesh.dim)
# check normals
gfnv = GridFunction(VectorH1(mesh,order=1))
gfnv.Set(specialcf.normal(2),definedon=mesh.Boundaries(".*"))
#Draw(gfnv, mesh, "nv")
#
print('Waveguide solution')
if Dirichlet:
    print('Solving the Dirichlet problem')
    fes = H1(mesh, order=porder, complex=True, dirichlet="dirichlet|PML")
else:
    print('Solving the penetrable problem')
    fes = H1(mesh, order=porder, complex=True, dirichlet="PML")
print(' ')
print('kappa=',kappa)
print('epsilon=',epsm)
print('Nres=',Nres)
print('collect:  Left=',collectL,'  right=',collectR)
print('Source position: x=',y1,' y=',y2[0],' to ',y2[Nres-1])
print('Wavelength in fluid: ',lam_f)
print('Wavelength in scatterer: ',lam_s)
print('PML distances=',pdm,pdp)
print('domain is [',-(delta+pdm),delta+pdp,']')
print('Labeled boundaries: ')
print(mesh.GetBoundaries())
print('Materials present: ')
print(mesh.GetMaterials())

mesh.SetPML(pml.HalfSpace(point=(pdp,H/2),normal=(1,0),alpha=4+2*1j),
                "PML_right")
mesh.SetPML(pml.HalfSpace(point=(-pdm,H/2),normal=(-1,0),alpha=4+2*1j),
                "PML_left")
kappa_fun=CoefficientFunction([kappa*np.sqrt(epsm) if mat=="scatterer" else kappa
                for mat in mesh.GetMaterials()])
scatter=CoefficientFunction([5 if mat=="scatter" else 1 if mat=="fluid" else 2
                if mat=="PML_right" else 3  if mat=="PML_left" else 4
                for mat in mesh.GetMaterials()])
Draw(scatter,mesh,'domains')
#Draw(kappa_fun,mesh,'Kappa')

u  = fes.TrialFunction()
v  = fes.TestFunction()

a=BilinearForm(fes)
a += SymbolicBFI(grad(u)*grad(v) - kappa_fun**2*u*v)
print('Number of DoFs: ',fes.ndof)

with TaskManager():
    a.Assemble()
    Ainv=a.mat.Inverse(freedofs=fes.FreeDofs(),inverse="umfpack")

ueval=np.zeros((Nres,Nres),dtype=complex)
amatL=np.zeros((Nterm+1,Nres),dtype=complex)
amatR=np.zeros((Nterm+1,Nres),dtype=complex)

beta_n=np.zeros(Nterm+1,dtype=complex)
for n in range(0,Nterm+1):
    beta_n[n]=np.emath.sqrt(kappa**2-(n*np.pi/H)**2)
    if np.imag(beta_n[n])==0.:
        next=n+1
        print('beta(',n,')=',beta_n[n])
if next <= Nterm:
    print('Next term beta(',next,')=',beta_n[next])
print('Final beta=',beta_n[Nterm])

def mode(m,yp):
    # returns transverse mode
    if m==0:
        xin=CoefficientFunction(np.sqrt(1/H)+0.*1j)
    else:
        xin=np.sqrt(2./H)*cos(m*np.pi*yp/H)+0.*1j
    return xin

for isv in range(0,Nres):
    if isv%5==0:
        print('Computing isv = ',isv,' of ',Nres-1)
    G=CoefficientFunction(0.+0.*1j) 
    for n in range(0,Nterm+1):
        xin_x=mode(n,y)
        xin_y=mode(n,y2[isv])
        if y1 < 0:
            G += exp( 1j*beta_n[n]*(x-y1))/(2.*1J*beta_n[n])*xin_x*xin_y
        else:
            G += exp(-1j*beta_n[n]*(x-y1))/(2.*1J*beta_n[n])*xin_x*xin_y
    #Draw(G,mesh,'G')   
    gfu = GridFunction(fes)
    if Dirichlet:
        gfu.vec[:]=0+0*1J
        gfu.Set(-G,BND,definedon=mesh.Boundaries("dirichlet"))
    
    f = LinearForm(fes)
    if not Dirichlet:
        f += SymbolicLFI(kappa**2*(epsm-1.)*G*v,
                         definedon=mesh.Materials('scatterer'))
    with TaskManager():
        f.Assemble()
    if Dirichlet:
        with TaskManager():
            res = f.vec.CreateVector()
            res.data = f.vec - a.mat * gfu.vec
            gfu.vec.data += Ainv * res
    else:
        with TaskManager():
            gfu.vec.data += Ainv * f.vec
    
    Draw (gfu, mesh,'p^s')
    for n in range(0,Nterm+1):
        xin=mode(n,y)
        amatL[n,isv]=Integrate(xin*gfu,mesh,BND,definedon=
                                   mesh.Boundaries("collectL"),order=porder+1)
        amatR[n,isv]=Integrate(xin*gfu,mesh,BND,definedon=
                                   mesh.Boundaries("collectR"),order=porder+1)
        #print(Integrate(x*y,mesh,BND,definedon=mesh.Boundaries("collect")))
        #xim=CoefficientFunction(np.sqrt(2./H)*cos(5*np.pi*y/H)+0.*1j)
        #print(Integrate(xin*xim,mesh,BND,definedon=mesh.Boundaries("collect")))



if Dirichlet:
    file='DiscDir_k'+str(kappa)+'_R'+str(collectL)+'t'+str(collectR)+'S'+str(y1)+'.mat'
    #file='Disc_Dirichlet_R'+str(Rsc)+'_'+str(kappa)+'_'+'S'+str(y1)+'.mat'
    #file='TwoDisc_Dirichlet'+'_'+str(kappa)+'_'+'S'+str(y1)+'.mat'
else:
    #file='Disc_penetrable_'+str(kappa)+'_'+str(collectL)+'_'+str(collectR)+'S'+str(y1)+'.mat'
    #file='Disc2_penetrable_R'+str(Rsc)+'_'+str(kappa)+'_'+'S'+str(y1)+'.mat'
    #file='TwoDisc_penetrable'+'_'+str(kappa)+'_'+'S'+str(y1)+'.mat'
    file='DiscLp_k'+str(kappa)+'_'+'S'+str(y1)+'.mat'
    #file='DiscRp_k'+str(kappa)+'_'+'S'+str(y1)+'.mat'
    #file='Disc2p_k'+str(kappa)+'_'+'S'+str(y1)+'.mat'
    #file='DiscCp_R'+str(Rsc)+'_'+'S'+str(y1)+'.mat'
    #file='DiscCp_y2'+str(y2sc)+'_'+'S'+str(y1)+'.mat'
    #file='DiscCp_'+'S'+str(y1)+'.mat'
    #file='DiscCp_'+'S'+str(y1)+'.mat'
    
sio.savemat(file,{'H':H,'Rsc1':Rsc1,'y1sc1':y1sc1,'y2sc1':y2sc1,
                      #'Rsc2':Rsc2,'y1sc2':y1sc2,'y2sc2':y2sc2,
                      'kappa':kappa,'Nres':Nres,'y1':y1,'y2':y2,'ueval':ueval,
                      'epsm':epsm,'Dirichlet':Dirichlet,'Nterm':Nterm,
                      'amatL':amatL,'amatR':amatR,'collectL':collectL,
                      'collectR':collectR,'betan':beta_n})
print('Results saved to ',file)
