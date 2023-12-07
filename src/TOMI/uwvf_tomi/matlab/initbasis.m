function basis = initbasis_FS(g,H,NE,f,E,nu,rhoS,rhoF,c,maxcond,indF,indS,A)

% Defines the number of plane wave basis functions for each element
% based on the conditioning of the matrix D in the 2-D UWVF
% for the coupled fluid-solid wave problem. The ratio of P- and
% S-wave basis functions in the solid part is k_P/k_S.
%
% NOTE: This version assumes that the domain consists of one fluid
% and one solid subdomain only !!!
%
%
% call: P = initbasis_FS(g,H,NE,f,E,nu,rhoS,rhoF,c,maxcond,indF,indS,A)
%
% input: 
%
% g     = (x,y)-coordinates of the vertices (n,2)
% H     = triangulation (m,4): in columns 1-3 are indeces to g
%                            : in column  4   are the indeces of 
%                            : the subdomains
% NE    = neighboring elements: n:th row includes indeces to 
%                               neighboring elements of the n:th
%                               element in H. NaN if on the boundary
% f     = frequency in Hz
% E     = Young's modulus for  the solid subdomain
% nu    = Poisson ratio for the solid  subdomain 
% rhoS  = density in kg/m^3 for the solid  subdomain 
% rhoF  = density in kg/m^3 for the fluid  subdomain 
% c     = speed of sound for the fluid  subdomain  
% maxcond = maximum allowed condition number of D_k
% indF    = subdomain index for  the fluid 
% indS    = subdomain index for  the solid
% A       = scaling factor for acoustic basis
%
% Output:
%
% basis = basisstruct:
%         basis.pP(i)   = number of P-wave bases for i:th element
%         basis.pS(i)   = number of S-wave bases for i:th element 
%                         (0 for fluid elements) 
%         basis.d(i).aP = P-wave directions for i:th element (basis.pP(i),2)
%         basis.d(i).aS = S-wave directions for i:th element
%                         (basis.pS(i),2) (empty for fluid elements) 


omega  = 2*pi*f;                   %angular frequency    
                                   
% fluid
kappa = omega/c; % wave number 

% solid
lambda = E*nu/((1+nu)*(1-2*nu));   %Lame constant 1
mu     = 0.5*E/(1+nu);             %Lame constant 2
c_P    = sqrt((lambda+2*mu)/rhoS); %speed of pressure wave 
c_S    = sqrt(mu/rhoS);            %speed of shear wave
k_P    = omega/c_P;                %pressure wave number
k_S    = omega/c_S;                %shear wave number


% number of elements 
NH = length(H(:,1));

% ratio of P- and S-wave basis functions
pdiv = k_S/k_P;

% the initial basis
if length(pdiv)<NH
  pdiv = pdiv*ones(NH,1);
end
basis.pP = 2*ones(NH,1);
basis.pS = round(pdiv.*2.*ones(NH,1));

%wait = waitbar(0,'Initializing basis');

% loop through all elements
for k = 1:NH

  %waitbar(k/NH,wait);
 
  condDk = 0;
  
  %  FLUID ELEMENT
  
  if H(k,4) == indF
    
    
    basis.pS(k) = 0;
    
    while condDk<maxcond
      
      basis.pP(k) = basis.pP(k) + 1;
      
      dsp=zeros([],2);
      
      % directions for the plane wave basis
      for ipP = 1:basis.pP(k)
        
        thetap=2*(ipP-1)*pi/basis.pP(k);
        dsp=[dsp;cos(thetap) sin(thetap)];
        
      end 
      
      basis.d(k).aP = dsp;
      
      % for simplicity, rename parameters 
      % associated with this element
      
      H_k    = H(k,1:3);
      
      p      = basis.pP(k);
      ds     = basis.d(k).aP;
      
      Dk = zeros(p,p);
     
      for n=1:3         

        for l=1:p

          for m=1:p
            
            % boundary parameter
            sigma=real(kappa/rhoF);
            
            % vertices for the face
            if n==3,
              x1=g(H_k(3),:);          
              x2=g(H_k(1),:);
            else
              x1=g(H_k(n),:);
              x2=g(H_k(n+1),:);
            end
        
            x21 = x2-x1;      
            % length of the face                 
            L = norm(x21);  
        
            % normal 
            nor = [x21(2) -x21(1)];
            nor = nor./norm(nor);
            % centroid of the element
            cent = sum(g(H_k,:))./3;
            % outward normal
            if dot(nor,(x2+x1)./2-cent)<0
              nor = -nor;
            end
        	  
		 
            ds_lm=ds(m,:)-ds(l,:);
            Z=exp(i*kappa*ds_lm*x1');    
            h=(kappa*ds_lm*x21')/2;
                
            if abs(h)<sqrt(6*eps)
              expsinhh=1;
            else
              expsinhh=exp(i*h)*sin(h)/h;
            end

            Dk(l,m)=Dk(l,m)+(1/sigma)*L*Z*...
                     (sigma+kappa/rhoF*nor*ds(m,:)')*...
                     (sigma+kappa/rhoF*nor*ds(l,:)')*expsinhh;
            
          end
		
        end 
    
      end
   
      Dk = A^2*Dk;
      
      condDk = cond(Dk);
      clear Dk
     
      
    end
    
    clear condDk
  
    % since the last value of condDk was above maxcond
    % we must go back one step
  
    basis.pP(k) = basis.pP(k) - 1;
    
    dsp=zeros([],2);
    
    % directions for the basis  
    for ipP = 1:basis.pP(k)
      
      thetap=2*(ipP-1)*pi/basis.pP(k);
      dsp=[dsp;cos(thetap) sin(thetap)];
     
    end 
   
    basis.d(k).aP = dsp;
    basis.d(k).aS = [];
   
  end
  
  
  % SOLID ELEMENT
  
  
  if H(k,4) == indS
  
    
    while condDk<maxcond
  
      basis.pP(k) = basis.pP(k) + 1;
      basis.pS(k) = round(pdiv(k)*basis.pP(k));
      
      dsp=zeros([],2);
      dss=zeros([],2);
      
      % directions for P-wave 
      for ipP = 1:basis.pP(k)
        
        thetap=2*(ipP-1)*pi/basis.pP(k);
        dsp=[dsp;cos(thetap) sin(thetap)];
        
      end 
      
      basis.d(k).aP = dsp;
   
      % directions for S-wave
      for ipS = 1:basis.pS(k)  
        
        thetas=2*(ipS-1)*pi/basis.pS(k);
        dss=[dss;cos(thetas) sin(thetas)];
        
      end 
      
      basis.d(k).aS = dss;
      
      % for simplicity, rename parameters 
      % for this element
      
      H_k    = H(k,1:3);
      
      pP     = basis.pP(k);
      pS     = basis.pS(k);
      a_P    = basis.d(k).aP;
      a_S    = basis.d(k).aS; 
      b_S    = [-a_S(:,2) a_S(:,1)]; 
      
      % the empty blocks
      D1 = zeros(pP,pP);
      D2 = zeros(pP,pS);
      D3 = zeros(pS,pP);
      D4 = zeros(pS,pS);
      
      % loop through edges
      for n=1:3
        
        if n==3,
          x1=g(H_k(3),:);          
          x2=g(H_k(1),:);
        else
          x1=g(H_k(n),:);
          x2=g(H_k(n+1),:);
        end
        
        x21 = x2-x1;      
        % length of the edge                   
        L = norm(x21);  
        % reshape for matrix operations
        x1P  = repmat(x1,pP,1);
        x1S  = repmat(x1,pS,1);
        x21P = repmat(x21,pP,1);
        x21S = repmat(x21,pS,1);
        
        % normal 
        nor = [x21(2) -x21(1)];
        nor = nor./norm(nor);
        % centroid of the element
        cent = sum(g(H_k,:))./3;
        % outward normal
        if dot(nor,(x2+x1)./2-cent)<0
          nor = -nor;
        end
        % tangent
        tng = [nor(2) -nor(1)];
        
        % as a matrix
        norP = repmat(nor,pP,1);
        norS = repmat(nor,pS,1); 
        tngS = repmat(tng,pS,1); 
        
        
        % subdomain for the adjacent element  
        if isnan(NE(k,n))==1 | NE(k,n)==0
          sda = H(k,4);
        else
          sda = H(NE(k,n),4);       
        end     
        
        % coupling parameter
        % adjacent element in the solid
        % i.e. solid-solid interface
        if sda == indS
       
          sgm = omega*rhoS*(c_P*nor'*nor+c_S*tng'*tng);
          
        % solid-fluid interface
        else
          
          sigma = real(kappa/rhoF);
          sgmP  = omega^2/sigma*(nor'*nor);
          sgmS  = omega*rhoS*c_S*(tng'*tng);
          %sgmS = (tng'*tng);
          sgm   = sgmP + sgmS;
        
          
        end
       
        
        % the dot products
        TPk = i*k_P*(2*mu*repmat(dot(norP,a_P,2),1,2).*a_P+lambda*norP);
        TSk = i*k_S*(2*mu*repmat(dot(norS,a_S,2),1,2).*b_S+mu*tngS);
        
        iPk = i*(sgm*a_P')';
        iSk = i*(sgm*b_S')';
        
        TT1 = conj(-TPk-iPk)*(inv(sgm)*(-TPk-iPk).');
        TT2 = conj(-TPk-iPk)*(inv(sgm)*(-TSk-iSk).');
        TT3 = conj(-TSk-iSk)*(inv(sgm)*(-TPk-iPk).');
        TT4 = conj(-TSk-iSk)*(inv(sgm)*(-TSk-iSk).');
        
        % the integral parts
        
        %block 1   
        tmp1 = k_P*dot(x1P,a_P,2);
        tmp2 = k_P*dot(x1P,a_P,2);
        Z  = exp(i*(repmat(tmp1.',pP,1)-repmat(tmp2,1,pP)));
        
        tmp1 = k_P*dot(x21P,a_P,2);
        tmp2 = k_P*dot(x21P,a_P,2);
        h    = (repmat(tmp1.',pP,1)-repmat(tmp2,1,pP))./2;
        expsinhh = zeros(size(h));
        ih = find(abs(h)<sqrt(6*eps));
        expsinhh(ih) = 1;
        ih = find(abs(h)>=sqrt(6*eps));
        expsinhh(ih)=exp(i*h(ih)).*sin(h(ih))./h(ih);
        
        D1 = D1 + TT1.*L.*Z.*expsinhh;
        
        %block 2   
        tmp1 = k_S*dot(x1S,a_S,2);
        tmp2 = k_P*dot(x1P,a_P,2);
        Z  = exp(i*(repmat(tmp1.',pP,1)-repmat(tmp2,1,pS)));
        
        tmp1 = k_S*dot(x21S,a_S,2);
        tmp2 = k_P*dot(x21P,a_P,2);
        h    = (repmat(tmp1.',pP,1)-repmat(tmp2,1,pS))./2;
        expsinhh = zeros(size(h));
        ih = find(abs(h)<sqrt(6*eps));
        expsinhh(ih) = 1;
        ih = find(abs(h)>=sqrt(6*eps));
        expsinhh(ih)=exp(i*h(ih)).*sin(h(ih))./h(ih);
        
        D2 = D2 + TT2.*L.*Z.*expsinhh; 

        %block 3   
        tmp1 = k_P*dot(x1P,a_P,2);
        tmp2 = k_S*dot(x1S,a_S,2);
        Z  = exp(i*(repmat(tmp1.',pS,1)-repmat(tmp2,1,pP)));
        
        tmp1 = k_P*dot(x21P,a_P,2);
        tmp2 = k_S*dot(x21S,a_S,2);
        h    = (repmat(tmp1.',pS,1)-repmat(tmp2,1,pP))./2;
        expsinhh = zeros(size(h));
        ih = find(abs(h)<sqrt(6*eps));
        expsinhh(ih) = 1;
        ih = find(abs(h)>=sqrt(6*eps));
        expsinhh(ih)=exp(i*h(ih)).*sin(h(ih))./h(ih);
        
        D3 = D3 + TT3.*L.*Z.*expsinhh; 

        %block 4   
        tmp1 = k_S*dot(x1S,a_S,2);
        tmp2 = k_S*dot(x1S,a_S,2);
        Z  = exp(i*(repmat(tmp1.',pS,1)-repmat(tmp2,1,pS)));
        
        tmp1 = k_S*dot(x21S,a_S,2);
        tmp2 = k_S*dot(x21S,a_S,2);
        h    = (repmat(tmp1.',pS,1)-repmat(tmp2,1,pS))./2;
        expsinhh = zeros(size(h));
        ih = find(abs(h)<sqrt(6*eps));
        expsinhh(ih) = 1;
        ih = find(abs(h)>=sqrt(6*eps));
        expsinhh(ih)=exp(i*h(ih)).*sin(h(ih))./h(ih);
        
        D4 = D4 + TT4.*L.*Z.*expsinhh; 
        
      end
      
      Dk =  [D1 D2; D3 D4];    
      
      condDk = cond(Dk);
     
      clear Dk
     
    end       
  
   
    clear condDk
  
    % since the last value of condDk was above maxcond
    % we must go back one step
  
    basis.pP(k) = basis.pP(k) - 1;
    basis.pS(k) = round(pdiv(k)*basis.pP(k));
    
    dsp=zeros([],2);
    dss=zeros([],2);
    
    % directions for P-wave 
    for ipP = 1:basis.pP(k)

      thetap=2*(ipP-1)*pi/basis.pP(k);
      dsp=[dsp;cos(thetap) sin(thetap)];
     
    end 
   
    basis.d(k).aP = dsp;
   
    % directions for S-wave
    for ipS = 1:basis.pS(k)  
  
      thetas=2*(ipS-1)*pi/basis.pS(k);
      dss=[dss;cos(thetas) sin(thetas)];

    end 
  
    basis.d(k).aS = dss;
  
   
  end

end

%close(wait);
