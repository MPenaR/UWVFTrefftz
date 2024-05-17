function y=thetan(x2,n,H)
    if n==0
        y=sqrt(1/H)+0.*x2;
    else
        y=sqrt(2/H)*cos(n*pi*x2/H);
    end
end