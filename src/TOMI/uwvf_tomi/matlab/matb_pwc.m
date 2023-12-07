function b=matb_pwc(g,H,NE,B,kappa,rho,d,Qs,basis,Amps)


% Computes the right hand side  for  the 2-D acoustic  UWVF and
% plane wave incidence over a  curved, circular  exterior boundary. 
%
%
% call: b=matb_pw(g,H,NE,kappa,rhoF,d)
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
% kappa = wave number
% rho   = density
% d     = direction of propagation pf the plane wave
% Qs    = the UWVF boundary parameters
% basis = a structure including information about the basis
%         basis.pP(i)   = number of P-wave bases for i:th element
%         basis.pS(i)   = number of S-wave bases for i:th element 
%                         (0 for fluid elements) 
%         basis.d(i).aP = P-wave directions for i:th element (basis.pP(i),2)
%         basis.d(i).aS = S-wave directions for i:th element
%                         (basis.pS(i),2) (empty for fluid elements) 
% A     = scaling factor for acoustic basis
% Amps  = Amplitude of the incident field for each boundary
%
% Output:
%
% b       = right hand side

[ib,jb]=find(isnan(NE)==1 | NE==0); 
Nb=length(ib);

b=sparse(sum(basis.pP),1);
sigma=real(kappa/rho);

for ii=1:Nb

  Hk = H(ib(ii),1:3);
  B_kn = B(ib(ii),jb(ii))
  Q = Qs(B_kn);
  Amp = Amps(B_kn)
  
  if ib(ii)==1
    istart = 1;
  else  
    istart = sum(basis.pP(1:ib(ii)-1)) + 1;
  end
  
  if jb(ii)==3
    x1=g(Hk(3),:);
    x2=g(Hk(1),:);
  else       
    x1=g(Hk(jb(ii)),:);
    x2=g(Hk(jb(ii)+1),:);
  end	
  
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
  
  p  = basis.pP(ib(ii));
  ds = basis.d(ib(ii)).aP;
  
  for l=1:p

    int_f=trapez_bpw(x1p,x2p,ds(l,:),kappa,rho,sigma,d,Q,B_kn);
    
    b(istart+l-1) = Amp*int_f;
    

  end

  
end


