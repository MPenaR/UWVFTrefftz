function int_f=trapez_bpw(x1p,x2p,dsl,kappa,rho,sigma,d,Q,bind)

% Computes the UWVF boundary integral for the right hand side  over a
% curved, circular exterior boundary and plane wave incidence. 
% Uses 64 point Gauss-Legendre.
%
% Call:  int_f=trapez_bpw(x1p,x2p,dsl,kappa,rho,sigma,d)
%
% Input:
%
% x1p   = 1st point in polar coordinates
% x2p   = 2nd point in polar coordinates
% dsl   = direction of plane wave test function
% kappa = wave number
% rho   = density
% sigma = coupling parameter
% d     = direction of the incident plane wave 
% Q     = the UWVF boundary parameter 
% bind  = boundary type index
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

x1=r*cos(theta);
x2=r*sin(theta);

x=[x1 x2];

if bind ==  2
  nor=[cos(theta) sin(theta)];
else
  nor=[-cos(theta) -sin(theta)];
end

dslv = repmat(dsl,64,1);
dv   = repmat(d,64,1);

int_f = 1/sigma*((1-Q)*sigma-(1+Q)*kappa/rho*dot(nor,dv,2)).*...
          (sigma-kappa/rho*dot(nor,dslv,2)).*...
            exp(i*dot(kappa*dv-conj(kappa)*dslv,x,2));


int_f = sum(int_f*r.*wght);
