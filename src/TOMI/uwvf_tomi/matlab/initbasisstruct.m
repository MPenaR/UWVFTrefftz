function basis = initbasisstruct(P)

% Builds a structure that includes basis function
% information for the 2D elastic UWVF.
%
% Call:  basis = initbasisstruct(P)
%
% Input: 
%
% P = (m,2)-matrix, where m is the number of elements:
%     first column includes number of bases for P-waves,
%     second for S-waves
%
% Output:
%
% basis = a struct where basis.pP (m,1) is the first column of P
%                        basis.pS (m,1) is the second column of P
%			 basis.d.aP(i)  (basis.pP(i),2) includes 
%                                       directions of propagation
%			                for P-waves in i:th element
%                        basis.d.aS(i)  same as above for S-waves	    


nt = length(P);

basis.pP = P(:,1);
basis.pS = P(:,2);
   
for it = 1:nt
   
   dsp=zeros([],2);
   dss=zeros([],2);

   % directions for P-wave 
   for ipP = 1:P(it,1)

     thetap=2*(ipP-1)*pi/P(it,1);
     dsp=[dsp;cos(thetap) sin(thetap)];
     
   end 
   
   basis.d(it).aP = dsp;
   
   % directions for S-wave
   for ipS = 1:P(it,2)  
  
     thetas=2*(ipS-1)*pi/P(it,2);
     dss=[dss;cos(thetas) sin(thetas)];

   end 
 
   basis.d(it).aS = dss;

end

  
