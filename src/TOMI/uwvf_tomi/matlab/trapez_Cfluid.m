function int_f=trapez_Cfluid(x1p,x2p,dsm,dsl,param,sigma,bind,sdind)

% Computes the UWVF boundary integral for the matrix C over a
% curved, circular exterior boundary and fluid-fluid interface. 
% Uses 64 point Gauss-Legendre
%
% Call:
% int_f=trapez_Cfluid(x1p,x2p,dsm,dsl,param,sigma,bind)
%
% Input:
%
% x1p   = 1st point in polar coordinates
% x2p   = 2nd point in polar coordinates
% dsm   = m'th direction of plane wave basis function in elem. j 
% dsl   = l'th direction of plane wave test function in elem. k
% param = vector for material paramaters: 
%         param(n,1) : density of fluid in domain n, rhoF
%         param(n,2) : wave number for fluid in domain n, kappa
%
% sigma = coupling parameter
% bind  = boundary index: 1 = exterior boundary
%                         2 = fluid-fluid boundary
% sdind = subdomain index 
%
% Output:
%
% int_f = boundary integral
%


r=x1p(1);
theta1=x1p(2);
theta2=x2p(2);

thetas = sort([theta2 theta1]);
theta1 = thetas(1);
theta2 = thetas(2);

[theta,wght] = gausslegendre64(theta1,theta2);
theta=theta';
wght=wght';

x=[r*cos(theta) r*sin(theta)]; 

nor=[cos(theta) sin(theta)];

dsmv = repmat(dsm,64,1);
dslv = repmat(dsl,64,1);


if bind == 2 | bind == 3
  
  if bind==3 nor=-nor; end
  
  % intergal for the exterior boundary
    
  rhok   = param(1,1);
  kappak = param(1,2);
 
  int_f=1/sigma*(sigma+kappak/rhok*dot(nor,dsmv,2)).*...
        (sigma-kappak/rhok*dot(nor,dslv,2)).*...
          exp(i*dot(kappak*(dsmv-dslv),x,2));

  
elseif bind == 1
  
  % integral for the curved fluid-fluid boundary
  
  if sdind==2 nor=-nor; end
  
  rhok   = param(1,1);
  kappak = param(1,2);
  rhoj   = param(2,1);
  kappaj = param(2,2);
  
 
  int_f=1/sigma*(sigma-kappaj/rhoj*dot(nor,dsmv,2)).*...
        (sigma-kappak/rhok*dot(nor,dslv,2)).*...
          exp(i*dot(kappaj*dsmv-kappak*dslv,x,2));
  
  
  
  
end
  

int_f = sum(int_f*r.*wght);
