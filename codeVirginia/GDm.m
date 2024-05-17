function y=GDm(s,H,N_mod,dx1y1,x2,y2)
% in a waveguide
    y=0.*bsxfun(@times,s,dx1y1);
    for n=0:N_mod
        ibn=ibetan_wsign(s,n,H);
        y=y-bsxfun(@rdivide, bsxfun(@times,exp(bsxfun(@times,ibn,abs(dx1y1))),...
                      bsxfun(@times,thetan(x2,n,H),thetan(y2,n,H))), ibn);
        % ATTENTION: the abs in abs(dx1y1) matters when sources 
        %            are placed on the right of the target...
    end
    y=y./2;
%
%     % in free space 
%     RY = bsxfun(@minus,x2,y2);
%     R = sqrt(dx1y1.^2+RY.^2);  
%     y = i/4*besselh(0,1,i*s*R);
% %
end
