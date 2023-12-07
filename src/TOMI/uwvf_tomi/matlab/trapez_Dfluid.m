function int_f=trapez_Dfluid(x1p,x2p,ds_m,ds_l,kappa,rho,sigma,bind,sdind)

% Computes the UWVF boundary integral for the matrix D over a
% curved interface on the fluid domain. Uses 64 point Gauss-Legendre
%
% Call:
% int_f=trapez_Dfluid(x1p,x2p,ds_m,ds_l,kappa,rho,sigma,bind)
%
% Input:
%
% x1p   = 1st point in polar coordinates
% x2p   = 2nd point in polar coordinates
% ds_m  = direction of plane wave basis function
% ds_l  = direction of plane wave test function
% kappa = wave number
% rho   = density
% sigma = coupling parameter
% bind  = boundary index: 2 = exterior boundary
%                         1 = fluid-fluid boundary
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


if bind == 2 
  nor=[cos(theta) sin(theta)];
elseif bind == 1
  if sdind == 1
    nor=[cos(theta) sin(theta)];
  else
    nor=[-cos(theta) -sin(theta)];
  end
elseif  bind == 3
  nor=[-cos(theta) -sin(theta)];
end

x=[r*cos(theta) r*sin(theta)]; 

int_f=1/sigma*(sigma+kappa/rho*nor*ds_m').*...
      (sigma+kappa/rho*nor*ds_l').*exp(i*(kappa*x*(ds_m-ds_l).'));


int_f=sum(int_f*r.*wght);
