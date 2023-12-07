function Cs = matCc(g,H,NE,B,f,rhoF,c,basis,Qs)

% Computes the matrix C in the 2-D UWVF for the acoustic
% wave  problem using curved, circular fluid-fluid interface and
% exterior boundary.
%
%
% call: Cs = matCc(g,H,NE,f,rhoF,c,basis,Q)
%
% input:
%
% g     = (x,y)-coordinates of the vertices (n,2)
% H     = triangulation (m,4): in columns 1-3 are indeces to g
%                            : in column  4   are the indeces of
%                            : the subdomains
% NE    = neighboring elements: n:th row includes indeces to
%                               neighboring elements of the n:th
%                              element in H. NaN if on the
%                               boundary
% B     = boundary index matrix
% f     = frequency in Hz
% c     = speed of sound for each subdomain
% basis = a structure including information about the basis
%         basis.pP(i)   = number of P-wave bases for i:th element
%         basis.pS(i)   = number of S-wave bases for i:th element
%                         (0 for fluid elements)
%         basis.d(i).aP = P-wave directions for i:th element (basis.pP(i),2)
%         basis.d(i).aS = S-wave directions for i:th element
%                         (basis.pS(i),2)  (empty for fluid elements)
% Qs     = boundary parameters for the exterior boundary
%
% Output:
%
% Cs     = the matrix for the UWVF
%

omega  = 2*pi*f;                   %angular frequency

% fluid
kappa = omega./c; % wave number


NH = length(H(:,1));

NC = 0;
nz = 0;
C  = [];

%wait = waitbar(0,'Computing C');

% loop through all elements
for k = 1:NH
    
    %waitbar(k/NH,wait);
    
    H_k=H(k,1:3);
    NE_k=NE(k,:);
    
    kappak = kappa(H(k,4));
    rhok   = rhoF(H(k,4));
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %  FLUID ELEMENT
    
    pk     = basis.pP(k);
    dsk    = basis.d(k).aP;
    
    NC = NC + pk;
    
    % row indeces for this block
    if k==1
        istart = 1;
        iend   = pk;
    else
        istart = iend + 1;
        iend   = iend + pk;
    end
    
    
    Ck = zeros(pk);
    
    for n=1:3
        
        % index of the adjacent element
        neig = NE(k,n);
        
        % boundary type index
        B_kn = B(k,n);
        
        if n==3,
            x1=g(H_k(3),:);
            x2=g(H_k(1),:);
        else
            x1=g(H_k(n),:);
            x2=g(H_k(n+1),:);
        end
        
        
        % number of quadrature points
        
        
        % ON THE CURVED EXTERIOR BOUNDARY
        if  B_kn == 2 | B_kn == 3% isnan(neig)==1 | neig==0
            
            Q = Qs(B_kn);
            
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
            
            
            sigma=real(kappak/rhok);
            
            
            for l=1:pk
                
                for m=1:pk
                    
                    int_f=trapez_Cfluid(x1p,x2p,dsk(m,:),dsk(l,:),...
                        [rhok kappak],sigma,B_kn,H(k,4));
                    
                    Ck(l,m) =  Ck(l,m) +  Q*int_f;
                    
                end
                
            end
            
            
            % ON THE STRAIGHT EXTERIOR BOUNDARY
        elseif  B_kn == 4
            
            Q = Qs(B_kn);
            
            sigma=real(kappak/rhok);
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
            
            for l=1:pk
                
                for m=1:pk
                    
                    dslm=kappak*dsk(m,:)-kappak*dsk(l,:);
                    Z=exp(i*dslm*x1');
                    h=(dslm*x21')/2;
                    
                    if abs(h)<sqrt(6*eps)
                        expsinhh=1;
                    else
                        expsinhh=exp(i*h)*sin(h)/h;
                    end
                    
                    Ck(l,m) =   Ck(l,m) +  ...
                        (Q/sigma)*(sigma+kappak/rhok*dot(nor,dsk(m,:)))*...
                        (sigma - kappak/rhok*dot(nor,dsk(l,:)))*L*Z*expsinhh;
                    
                    
                end
                
            end
            
            
            % ON THE INTERNAL  INTERFACE
        else
            
            sda  = H(NE(k,n),4);
            
            kappaj = kappa(sda);
            rhoj   = rhoF(sda);
            
            sigma = 0.5*(real(kappak)/rhok+real(kappaj)/rhoj);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            % ON THE STRAIGHT  FLUID-FLUID INTERFACE
            if B_kn == 0 %sda == H(k,4)
                
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
                
                pj     = basis.pP(neig);
                dsj    = basis.d(neig).aP;
                
                % column indeces for this block
                if neig==1
                    jstart = 1;
                    jend   = pj;
                else
                    jstart = sum(basis.pP(1:neig-1))+1;
                    jend   = jstart + pj - 1;
                end
                
                Ckj = zeros(pk,pj);
                
                for l=1:pk
                    
                    for m=1:pj
                        
                        dslm=kappaj*dsj(m,:)-kappak*dsk(l,:);
                        Z=exp(i*dslm*x1');
                        h=(dslm*x21')/2;
                        
                        if abs(h)<sqrt(6*eps)
                            expsinhh=1;
                        else
                            expsinhh=exp(i*h)*sin(h)/h;
                        end
                        
                        Ckj(l,m) =  ...
                            (1/sigma)*(sigma-kappaj/rhoj*dot(nor,dsj(m,:)))*...
                            (sigma - kappak/rhok*dot(nor,dsk(l,:)))*L*Z*expsinhh;
                        
                        
                    end
                    
                end
                
                nz = nz + pk*pj;
                
                [cind, rind] = meshgrid(jstart:jend,istart:iend);
                C = [C; rind(:) cind(:) Ckj(:)];
                
            end
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            % ON THE CURVED FLUID-FLUID INTERFACE
            if B_kn == 1 %sda ~= H(k,4)
                
                %nodes in polar coordinates
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
                
                param = [rhok kappak; rhoj kappaj];
                
                
                pj     = basis.pP(neig);
                dsj    = basis.d(neig).aP;
                
                nz = nz + pk*pj;
                
                % column indeces for this block
                if neig==1
                    jstart = 1;
                    jend   = pj;
                else
                    jstart = sum(basis.pP(1:neig-1))+1;
                    jend   = jstart + pj -1;
                end
                
                % the empty blocks
                Ckj = zeros(pk,pj);
                
                
                for l=1:pk
                    
                    % P-wave basis
                    
                    for m=1:pj
                        
                        int_f=trapez_Cfluid(x1p,x2p,dsj(m,:),dsk(l,:),...
                            param,sigma,B_kn,H(k,4));
                        Ckj(l,m) = int_f;
                        
                    end
                    
                    
                end
                
                nz = nz + pk*pj;
                [cind, rind] = meshgrid(jstart:jend,istart:iend);
                C = [C; rind(:) cind(:) Ckj(:)];
                
                
            end
            
            
        end
        
        
    end
    
    % the diagonal block for elements on the
    % exterior boundary
    
    if sum(sum(abs(Ck)))>10*eps*sum(sum(abs(Ck)))
        nz = nz + (iend-istart+1)^2;
        [cind, rind] = meshgrid(istart:iend,istart:iend);
        C = [C; rind(:) cind(:) Ck(:)];
    end
    
    
end %loop through elements


% build the sparse C

Cs = sparse(C(:,1),C(:,2),C(:,3),NC,NC,nz);

%close(wait)
