%%%AN UWVF SOLUTION FOR SCATTERING%%%

clear 
close all
clc

% UWVF approximation is computed in this mesh

load ../data/mesh_sc26.mat;
g = 0.2*g;

H(:,4) = B(:,4);
B = B(:,1:3);

% plot mesh
figure,
plcigrid(g,H(:,1:3))


% results are interpolated at these point
NP = 101;
t=linspace(min(min(g)),max(max(g)),NP);
[XX,YY]=meshgrid(t,t);
nor=sqrt(XX.^2+YY.^2);
gnew = [XX(:) YY(:)];
% points gnew in polar coordinates 
[thg,rg] = cart2pol(gnew(:,1),gnew(:,2));
gpnew = [rg thg];


% physical parameters
f      = 120e3;      % frequency
omega  = 2*pi*f;     % angular frequency   
a = 0.01;            % radius of the scatter
R = 0.02;            % radius of the domain 

% material properties for each subdomain
rhoF = [1000]; % density
c    = [1500]; % speed of sound


% number of basis functions for aech element 
PP = zeros(length(H),2);
PP(:,1) = 11; 

% built basis structure
basis = initbasisstruct(PP);

% matrix D
tic,[D,Dinv,Dcond]=matDc(g,H,NE,B,f,rhoF,c,basis); toc

% matrix C
Q = [0 0 1];
tic,C = matCc(g,H,NE,B,f,rhoF,c,basis,Q); toc

M = Dinv*C;
IM = (speye(size(M))-M);
 

% direction of the incoming
theta=0;
d=[cos(theta) sin(theta)]; 

% amplitude for each boundary
Amp    = [0 0 -1];        
% right hand side
kappa = omega./c;
tic,b=matb_pwc(g,H,NE,B,kappa,rhoF,d,Q,basis,Amp); toc 
bp = Dinv*b;

% solve 
tic,X  = full(IM\bp); toc;


% interpolate field at points gnew
u = evalfield(X,g,H,f,c,basis,gnew);

% reshape for plot
U=reshape(u,NP,NP);

% set point r>R  NaN
r = sqrt(XX.^2+YY.^2);
U(find(r<a | r>R)) = NaN+i*NaN; 

% plot
figure(1)
subplot(2,2,1),
contourf(XX,YY,real(U))
%imagesc(t,t,real(UE)),
axis square,colorbar('vert')
title('UWVF, real');
subplot(2,2,2),
contourf(XX,YY,imag(U))
%imagesc(t,t,real(UE)),
axis square,colorbar('vert')
title('UWVF, imag');
subplot(2,2,3),
%imagesc(t,t,abs(UE)),%caxis([0 2]),
contourf(XX,YY,abs(U))
axis square,colorbar('vert'),
title('UWVF, abs');
subplot(2,2,4)
poly=zeros(length(H),1);
pltpoly(g,H,poly)
title('Mesh: Polynomial triangles are red')
colormap('default')


% EXACT SOLUTION 

% truncation tolerance of the Fourier solution 
tol = 1e-5;
Nmax = 100;

% solution of the physical scattering problem 
[ue,conve]=exacthard2(gpnew,kappa,a,tol,Nmax);


% plot
UE=reshape(ue,NP,NP);
UE(find(r<a | r>R)) = NaN+i*NaN; 

figure
subplot(2,2,1),
contourf(XX,YY,real(UE))
%imagesc(t,t,real(UE)),
axis square,colorbar('vert')
title('Exact physical, real');
subplot(2,2,2),
contourf(XX,YY,imag(UE))
%imagesc(t,t,real(UE)),
axis square,colorbar('vert')
title('Exact physical, real');
subplot(2,2,3),
%imagesc(t,t,abs(UE)),%caxis([0 2]),
contourf(XX,YY,abs(UE))
axis square,colorbar('vert'),
title('Exact physical, abs');
colormap('default')

% error

nonnan = find(isnan(U)==0);
err1 = 100*norm(U(nonnan)-UE(nonnan))/norm(UE(nonnan))


