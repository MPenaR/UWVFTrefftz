function [D,Dinv,Dcond]=matDc(g,H,NE,B,f,rhoF,c,basis)


% Computes the matrix D in the 2-D UWVF for the acoustic
% wave problem using curved, circular fluid-fluid interface and
% exterior boundary.
%
% NOTE: This version assumes that the domain consists of one fluid
% and one solid subdomain only !!!
%
%
% call: [D,Dinv]=matDc(g,H,NE,f,rhoF,c,basis)
%
% input: 
%
% g     = (x,y)-coordinates of the vertices (n,2)
% H     = triangulation (m,4): in columns 1-3 are indeces to g
%                            : in column  4   are the indeces of 
%                            : the subdomains
% NE    = neighboring elements: n:th row includes indeces to 
%                               neighboring elements of the n:th
%                               element in H. NaN if on the
%                               boundary
% B     = boundary index matrix    
% f     = frequency in Hz
% rhoF  = density in kg/m^3 for the fluid  subdomain 
% c     = speed of sound for the fluid  subdomain  
% basis = a structure including information about the basis
%         basis.pP(i)   = number of P-wave bases for i:th element
%         basis.pS(i)   = number of S-wave bases for i:th element 
%                         (0 for fluid elements) 
%         basis.d(i).aP = P-wave directions for i:th element (basis.pP(i),2)
%         basis.d(i).aS = S-wave directions for i:th element
%                         (basis.pS(i),2) (empty for fluid elements) 
%
% Output:
%
% D     = matrix D
% Dinv  = inverse of D
% Dcond = condition number of each matrix block D_k  
         
omega  = 2*pi*f;                   %angular frequency    
                                   
% fluid
kappa = omega./c; % wave number 

N_H  = length(H(:,1));
ND   = 0;
nz   = 0;
D    = [];

%wait=waitbar(0,'Matrix D');

for k=1:N_H
  
  
  %waitbar(k/N_H,wait)
  
  H_k  = H(k,1:3);
  NE_k = NE(k,:);
  
  rhok = rhoF(H(k,4));
  kappak = kappa(H(k,4));
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  %  FLUID ELEMENT
  
  
  p      = basis.pP(k);
  ds     = basis.d(k).aP;
  
  Dk=zeros(p,p);
  
  ND = ND + p;
  nz = nz + p^2;
  
  if k==1
    istart = 1;
    iend   = p;
  else   
    istart = iend + 1;
    iend   = iend + p;
  end
  
  for n=1:3 
    
    % boundary type index
    B_kn  = B(k,n); 
    
    sigma=real(kappak/rhok);
    
    if n==3,
      x1=g(H_k(3),:);
      x2=g(H_k(1),:); 
    else
      x1=g(H_k(n),:);
      x2=g(H_k(n+1),:);
    end
    
    
    % index of the adjacent element
    neig = NE(k,n);
    
    if isnan(neig)==0 & neig~=0
      % subdomain index of the adjacent element
      sda  = H(neig,4);
      rhoj = rhoF(sda);
      kappaj = kappa(sda);
      sigma  = 0.5*(real(kappak)/rhok+real(kappaj)/rhoj);
    end
    
    % ON THE CURVED EXTERIOR BOUNDARY OR
    % ON THE CURVED FLUID-FLUID INTEFACE (=curved faces)	
    if  B_kn==1 | B_kn==2 | B_kn==3 %isnan(neig)==1 | neig==0 | sda~=H(k,4)
      
      % nodes in polar coordinates 
      [th, r] = cart2pol(x1(1),x1(2));
      if th<0  th = th+2*pi; end
      x1p(1) = r;
      x1p(2) = th; 
      
      [th, r] = cart2pol(x2(1),x2(2));
      if th<0  th = th+2*pi; end
      x2p(1) = r;
      x2p(2) = th;
      
      % take into account 2*pi periodicity
      if abs(x2p(2)-x1p(2))>pi 
        if x2p(2)>pi x2p(2)=x2p(2)-2*pi; end
        if x1p(2)>pi x1p(2)=x1p(2)-2*pi; end
      end
      
      
      for l=1:p
        
        for m=1:p 
          
          int_f=trapez_Dfluid(x1p,x2p,ds(m,:),ds(l,:),...
                              kappak,rhok,sigma,B_kn,H(k,4));
          
          Dk(l,m)=Dk(l,m) + int_f;
          
        end
        
      end
      
      
      % ON THE INTERNAL  INTERFACE (=straight face)  
    else 
      
      x21=x2-x1;
      L=norm(x21);
      
      % normal 
      nor = [x21(2) -x21(1)];
      nor = nor./norm(nor);
      % centroid of the element
      cent = sum(g(H_k,:))./3;
      % outward normal
      if dot(nor,(x2+x1)./2-cent)<0
        nor = -nor;
      end
      
      for l=1:p
        
        for m=1:p 
          
          dslm=ds(m,:)-ds(l,:);
          Z=exp(i*kappak*dot(dslm,x1));    
          h=kappak*dot(dslm,x21)/2;
          
          if abs(h)<sqrt(6*eps)
            expsinhh=1;
          else
            expsinhh=exp(i*h)*sin(h)/h;
          end
          
          Dk(l,m)=Dk(l,m)+...
                  (1/sigma)*(sigma+kappak/rhok*dot(nor,ds(m,:)))*...
                  (sigma+kappak/rhok*dot(nor,ds(l,:)))*L*Z*expsinhh;
          
        end
        
      end 
      
    end
    
  end

  [cind, rind] = meshgrid(istart:iend,istart:iend);
  
  Dkinv    = inv(Dk);
  Dcond(k) = cond(Dk,1);
  
  D   = [D; rind(:) cind(:) Dk(:) Dkinv(:)]; 
  clear Dk Dkinv    
  
end

% build the sparse D
Dinv = sparse(D(:,1),D(:,2),D(:,4),ND,ND,nz);   
D    = sparse(D(:,1),D(:,2),D(:,3),ND,ND,nz);    


%close(wait)
