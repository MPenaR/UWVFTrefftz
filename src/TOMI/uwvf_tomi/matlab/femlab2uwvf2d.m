function [g,H,NE,NOR,B]=femlab2uwvf2d(fem,subdind,intfind,elabels)

% converts the output of the FEMLAB mesh generator 
% to suitable for the UWVF
%
% [g,H,NE,NOR,B]=femlab2uwvf2d(fem,subdind,intfind)
%
% input:
% fem = the FEMLAB fem structure including the fem.mesh cell
%
% (optional) sunbdind = ordering of the subdomains: first entry
%                       correspond to the subdomain labeled with 1
%                       in B, etc
% (optional) intfind  = ordering of the interfaces: the interface
%                       between subdomains of first row is labeled
%                       with 1 in B, etc
% (optional) elabels  = edgelabels, first column includes edgelabels
%                       in the FEMLAB mesh, second column includes
%                       corresponding labels in the output mesh.
%                       if an edgelabel is not in the first column, 
%                       the edge in the output is labeled as zero
%                          
%
% output:
% g = vertices
% H = triangulation
% NE = neighbor for each element
% NOR = outward normal for each edge
% B = boundary and subdomain indeces;


p = fem.mesh.p;
t = fem.mesh.t;
e = fem.mesh.e;

g=p';
H=t(1:3,:)';
e=e';
bs = [];

if nargin==1,

  % number of subdomains 
  nsd = max(t(4,:)); 

  for id = 1:nsd

    % find the index pairs for edges on this boundary
    sd1 = nsd-id;
    sd2 = nsd-id+1;
    ind = int2str(id);
    bs = setfield(bs,['b' ind], e(find((e(:,6)==sd1 & e(:,7)==sd2) | ...
				       (e(:,6)==sd2 & e(:,7)==sd1)),1:2));
    
  end
  
elseif nargin==3 
  
  % number of subdomains 
  nsd = length(subdind);
  
  for id = 1:nsd
    
    % find the index pairs for edges on this boundary
    sd1 = intfind(id,1);
    sd2 = intfind(id,2);
    ind = int2str(id);
    bs = setfield(bs,['b' ind], e(find((e(:,6)==sd1 & e(:,7)==sd2) | ...
				       (e(:,6)==sd2 & e(:,7)==sd1)),1:2));
    
    if  isempty(getfield(bs,['b' ind]))==1,
      disp('Something wrong in the interface labeling!!!')
    end  
    
   
  end
  
elseif  nargin==4
       epoints    = e(:,1:2);
       allelabels = e(:,5); 
      
else
  disp('Too many / too few inputs !!!')
  break;
  
  
end
  
  
N_H=length(H(:,1));
NE=zeros(N_H,3);
NOR=zeros(N_H,6);
B=sparse(N_H,4);

wait=waitbar(0,'Be patient, Please!');

for ii=1:N_H
	
   waitbar(ii/N_H)
  
   Hii=H(ii,:);
   tii=t(4,ii);

   %define the subdomain index for this element 
   
   if nargin==1,
     B(ii,4)=nsd-tii+1;
   else
     B(ii,4)=find(subdind==tii);
   end

   %loop through all edges
   
   for jj=1:3

      %find indeces for this edge n0 is the third index 
      %on this element
     
      if jj==1 
         n1=Hii(1);
         n2=Hii(2);
         n0=Hii(3);
      elseif jj==2
         n1=Hii(2);
         n2=Hii(3);
         n0=Hii(1);
      else
         n1=Hii(3);
         n2=Hii(1);
         n0=Hii(2);
      end

      %find normal for this edge
      
      pp1=g(n1,:);
      pp2=g(n2,:);
      pp0=g(n0,:); 

      a=pp1-pp0;
      b=pp2-pp1;
      
      NOR(ii,jj+(jj-1):2*jj)=[b(2) -b(1)]/norm([b(2) -b(1)]);  
 
      %find neighbour: NaN if on the free edge
      
      [in1,jn1]=find(H==n1);
      [in2,jn2]=find(H==n2);

      HH=intersect(in1,in2);
      
      if length(HH)==1
          NE(ii,jj)=NaN;
      else
          NE(ii,jj)=setdiff(HH,ii); 
      end                

      if nargin < 4
      
         %find if this edge is on the domain interface 
 
         for ib = 1:nsd
      
         %check boundary 
      
         bib = getfield(bs,['b' int2str(ib)]);
      
         [in1b,jn1b]=find(bib==n1);
         [in2b,jn2b]=find(bib==n2);
      
         inb=intersect(in1b,in2b);
      
         if isempty(inb)==0
  	    B(ii,jj)=ib;
	    clear in1b1 jn1b1 in2b1 jn2b1 ib1
	    break;
         end
      
         end
      
      else
	
	% find FEMLAB edgelabel (if exists)  for this edge
	[inb,iep,ip] = intersect(epoints,[n1 n2; n2 n1],'rows');
	label        = allelabels(iep);
	
	% change labeling in B if requested
	if sum(ismember(elabels(:,1),label))~=0
	  B(ii,jj) = elabels(find(elabels(:,1)==label),2);
	end
	 
      end

   end
end
    
close(wait)
