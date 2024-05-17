function y=ibetan_wsign(s,n,H)
    y=sqrt(s.^2+(n*pi/H).^2); 
    for j=1:length(y)
        if abs(real(y(j)))>10*eps
            y(j)=-sign(real(y(j)))*y(j);
        else
            y(j)=-sign(imag(s))*sign(imag(y(j)))*y(j);
        end
    end
%     ii = find(abs(real(y))>eps)
%     y(ii)=
%     if abs(real(y))>eps
%         y=-sign(real(y)).*y;
end