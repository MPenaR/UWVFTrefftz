function [usc,conve] = exacthard2(gp,k,r0,tol,N)

% Computes scattered field from a sound-hard cylinder.
%
% Call:  usc = exacthard2(g,gp,k,a,tol,N)
%
% Input:
%
% gp  = the grid in polar coordinates
% k   = wave number
% r0  = radius of the scatterer
% tol = truncation tolerance
% N   = maximum number of Fourier modes
%
% Output:
%
% usc   = scattered field
% conve = convergence


n  = 0:N;
th = gp(:,2)';
r  = gp(:,1);

% remove internal points
ind = find(r<r0);
r(ind)  = NaN+i*NaN;
th(ind) = NaN+i*NaN;
nonnan = setdiff(1:length(gp),ind);

a   = (besselj(n-1,k*r0)-besselj(n+1,k*r0))./...
      (besselh(n-1,1,k*r0)-besselh(n+1,1,k*r0));

% zeroth order mode
usc = -a(1)*besselh(0,1,k*r);
wait = waitbar(0,'Maximum expected time!!');

for m = 1:N
  
  waitbar(m/N)
  
  usc_ex = usc;
  
  ur  = besselh(m,1,k*r);
  uth = 2*i^m*cos(m*th)';

  usc = usc - a(m+1)*ur.*uth;

  conve(m) =  norm(usc_ex(nonnan)-usc(nonnan))/norm(usc(nonnan));
  
  if conve(m)<tol
    disp('Requested tolerance reached!')
    break;
  end
  
end

if m==N
 disp('Maximum number of Fourier modes reached!!') 
end

close(wait)
