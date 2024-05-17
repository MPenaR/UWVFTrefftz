function G=GD(s,Hwguide,N_im,dx1y1,dx2y2,dx2y2m)
%
%    % for Helmholtz eq...
%        D = sqrt(dx1y1.^2+dx2y2.^2);                    % |m_i^ep-m_j|
%        G = 1i/4.*besselh(0,1,1i*s*D);
% %
%     % % for a waveguide...
    G=0.*dx1y1;
    for j=-N_im:N_im
        % y displaced
        dx2y2_j= dx2y2 +2*j*Hwguide;
        rxy_j=sqrt(dx1y1.^2+dx2y2_j.^2);
        % y' displaced
        dx2y2m_j= dx2y2m +2*j*Hwguide;
        rxym_j=sqrt(dx1y1.^2+dx2y2m_j.^2);
        % "assembly"
        G=G+1i*(besselh(0,1,1i*s*rxy_j)+besselh(0,1,1i*s*rxym_j))./4;
     end  
%
end