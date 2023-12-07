function u = evalfield(X,g,H,f,c,basis,gnew)

% Evaluates Helmholtz   UWVF solution, computed in mesh (g,H), in points 
% gnew
%
% call: u = evalfield(X,g,H,f,c,basis,gnew)
%
% input: 
%
% X     = weights for basis functions
% g     = (x,y)-coordinates of the vertices (n,2)
% H     = triangulation (m,4): in columns 1-3 are indeces to g
%                            : in column  4   are the indeces of 
%                            : the subdomains
% f     = frequency in Hz
% c     = speed of sound for the fluid  subdomain  
% basis = a structure including information about the basis
%         basis.pP(i)   = number of P-wave bases for i:th element
%         basis.pS(i)   = number of S-wave bases for i:th element
%         basis.d(i).aP = P-wave directions for i:th element (basis.pP(i),2)
%         basis.d(i).aS = S-wave directions for i:th element
%         (basis.pS(i),2)      
% gnew  = a set of points in which the field is evaluated
%
% Output:
%
% u    =  field in points gnew
%
 

omega = 2*pi*f;
kappa = omega./c;

% find the enclosing triangle for the points gnew

enctri = tsearchn([g(:,1) g(:,2)],H(:,1:3),[gnew(:,1) gnew(:,2)]);

% maybe the point was on the edge (or outside of the domain)
% then find the nearest element

nanind = find(isnan(enctri)==1);
% neighboring node
nextnod = dsearchn([g(:,1) g(:,2)],H(:,1:3),[gnew(nanind,1) gnew(nanind,2)]);

NNan = length(nanind);

for j = 1:NNan

  [it,jt] = find(H(:,1:3)==nextnod(j));
 
  dist = [];
  for k = 1:length(it)
   
    centroid = (g(H(it(k),1),:)+g(H(it(k),2),:)+g(H(it(k),3),:))/3;
    dist(k) = norm(gnew(nanind(j),:)-centroid);
    
  end 
  
  xp(j,:)  = gnew(nanind(j),:);
  closest = find(dist==min(dist));
  enctri(nanind(j)) = it(closest);
    
end


Ng=length(gnew(:,1));
NH=length(H(:,1));

u=zeros(Ng,1);

for iv = 1:Ng 
  
  % enclosing triangle
  tri = enctri(iv);         
  kappak = kappa(H(tri,4));
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
  p     = basis.pP(tri);
  ds    = basis.d(tri).aP;
  
  % indeces for this triangle

  if tri==1
    istart = 1;
    iend   = p;
    
  else    
    istart = sum(basis.pP(1:tri-1))+1;
    iend   = istart + p - 1;   
  end
  
  % weights
  XP = X(istart:iend);
    
  
  for l=1:p
    
    u(iv) = u(iv) + ...
              XP(l)*exp(i*kappak*dot(ds(l,:),gnew(iv,:)));
    
  end
  

end



