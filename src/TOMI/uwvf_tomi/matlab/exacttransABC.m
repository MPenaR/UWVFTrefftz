function [conve,trs]=exacttransABC(gp,r1,r2,k1,k2,rho1,rho2,tol)


kmax=max([k1 k2]);

N=round(1.6*r2*kmax);

ind1=find(gp(:,1)<r1);
ind2=find(gp(:,1)>=r1);  
gp1=gp(ind1,:);
gp2=gp(ind2,:);
th1=gp1(:,2);
rr1=gp1(:,1);
th2=gp2(:,2);
rr2=gp2(:,1);
trs1=zeros(length(rr1),1);
trs2=zeros(length(rr2),1);
trs=zeros(length(gp(:,1)),1);

J1n=besselj(0,k1*r1);
H1n=-besselh(0,1,k2*r1);
H2n=-besselh(0,2,k2*r1);
J2n=besselj(0,k2*r1);


Jd1n=k1/rho1/2*(besselj(-1,k1*r1)-besselj(1,k1*r1));
Hd1n=-k2/rho2/2*(besselh(-1,1,k2*r1)-besselh(1,1,k2*r1));
Hd2n=-k2/rho2/2*(besselh(-1,2,k2*r1)-besselh(1,2,k2*r1));
Jd2n=k2/rho2/2*(besselj(-1,k2*r1)-besselj(1,k2*r1));

Ha=0.5*(besselh(-1,1,k2*r2)-besselh(1,1,k2*r2))-i*besselh(0,1,k2*r2);
Hb=0.5*(besselh(-1,2,k2*r2)-besselh(1,2,k2*r2))-i*besselh(0,2,k2* ...
						  r2);

b=(Jd2n-J2n*Jd1n/J1n)/(Ha/Hb*(H2n*Jd1n/J1n-Hd2n)+Hd1n-H1n*Jd1n/J1n);
a=J2n/J1n+1/J1n*(H2n*Ha/Hb-H1n)*b;
c=-Ha/Hb*b;

trs1 = a*besselj(0,k1*rr1);
trs2 = besselj(0,k2*rr2)+b*besselh(0,1,k2* rr2)+c*besselh(0,2,k2*rr2);

conve(1:2)=1;
m=1;

wait=waitbar(0,'Expected time');

while conve(m)>tol
    
    waitbar(m/N)
  
    J1n=besselj(m,k1*r1);
    H1n=-besselh(m,1,k2*r1);
    H2n=-besselh(m,2,k2*r1);
    J2n=i^m*besselj(m,k2*r1);


    Jd1n=k1/rho1/2*(besselj(m-1,k1*r1)-besselj(m+1,k1*r1));
    Hd1n=-k2/rho2/2*(besselh(m-1,1,k2*r1)-besselh(m+1,1,k2*r1));
    Hd2n=-k2/rho2/2*(besselh(m-1,2,k2*r1)-besselh(m+1,2,k2*r1));
    Jd2n=i^m*k2/rho2/2*(besselj(m-1,k2*r1)-besselj(m+1,k2*r1));

    Ha=0.5*(besselh(m-1,1,k2*r2)-besselh(m+1,1,k2*r2))-i*besselh(m,1,k2*r2);
    Hb=0.5*(besselh(m-1,2,k2*r2)-besselh(m+1,2,k2*r2))-i*besselh(m,2,k2* ...
						  r2);

    	
    if isfinite((Jd2n-J2n*Jd1n/J1n)/(Ha/Hb*(H2n*Jd1n/J1n-Hd2n)+Hd1n-H1n*Jd1n/J1n))==0 |...
	isfinite(J2n/J1n+1/J1n*(H2n*Ha/Hb-H1n))==0
	disp('ZERO NOMINATOR IN FOURIER SERIES!!!');
	disp(['number of fourier modes=' num2str(m)]);
	disp(['truncation error=' num2str(conve(m))]);
	break;
    end

    
    b=(Jd2n-J2n*Jd1n/J1n)/(Ha/Hb*(H2n*Jd1n/J1n-Hd2n)+Hd1n-H1n*Jd1n/J1n);
    a=J2n/J1n+1/J1n*(H2n*Ha/Hb-H1n)*b;
    c=-Ha/Hb*b;

   
  
  if m>=2
     trs_ex=trs;
  end 

  trr1 = a*besselj(m,k1*rr1);
  tth1 = 2*cos(m*th1);
  trs1 = trs1+trr1.*tth1;
 
  trr2 = (i)^m*besselj(m,k2*rr2)+b*besselh(m,1,k2*rr2)+c*besselh(m,2,k2*rr2);
  
  tth2 = 2*cos(m*th2);
  trs2 = trs2+trr2.*tth2;

  trs(ind1)=trs1;
  trs(ind2)=trs2;
  
  if m>=2
    conve(m+1)=norm(trs_ex-trs)/norm(trs);
  end
  m=m+1;
  
  
end

disp(['number of fourier modes=' num2str(m)]);
disp(['truncation error=' num2str(conve(m))]);
close(wait)
