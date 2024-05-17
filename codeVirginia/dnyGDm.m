function y=dnyGDm(s,H,N_mod,dx1y1,x2,y2,ny)
%
% in the waveguide
    y=0.*bsxfun(@times,s,dx1y1);
    y1aux=y; y2aux=y; %auxiliary variables... we could remove them!!!
    %
    for n=0:N_mod
        ibn=ibetan_wsign(s,n,H);
        y1aux=bsxfun(@times,thetan(y2,n,H), ny(:,1)) ;
        % y1aux=bsxfun(@times, thetan(y2,n,H),
        % bsxfun(@times,ny(:,1)),sign(dx1y1)) ; % THIS DOES NOT WORK!!!
        y2aux=bsxfun(@rdivide,bsxfun(@times,dthetan(y2,n,H),ny(:,2)),ibn);
        y=y-bsxfun(@times , bsxfun(@times,exp(bsxfun(@times,ibn,abs(dx1y1))),thetan(x2,n,H)) , ...
                    bsxfun(@plus,y1aux,y2aux) ) ;
        % ATTENTION: the abs in abs(dx1y1) matters when sources  are placed on the right of the target... 
        %            care about the deriv wrt y1?!?!?!
        %            UNDONE FOR SOURCES ON THE RIGHT!!! 
    end
    y=y./2;
%
%     % in free space
%     RY = bsxfun(@minus,x2,y2);
%     R = sqrt(dx1y1.^2+RY.^2);  
%     RN = bsxfun(@times,dx1y1,ny(:,1)) ...
%             +bsxfun(@times,RY,ny(:,2));
%     RN = RN./R;
%     y = -s/4*besselh(1,1,i*s*R).*RN;
%
%
end