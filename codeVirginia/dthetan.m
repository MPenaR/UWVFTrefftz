function y=dthetan(x2,n,H)
    if n==0
        y=0.*x2;
    else
        y=-(sqrt(2/H)*n*pi/H)*sin(n*pi*x2/H);
    end
end