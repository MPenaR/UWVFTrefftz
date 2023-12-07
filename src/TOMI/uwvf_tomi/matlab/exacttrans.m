function [conve,trs]=exacttrans(gp,r1,k1,k2,rho1,rho2,tol)

rmax=max(gp(:,1));
kmax=max([k1 k2]);

N=round(1.6*rmax*kmax);

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
J2n=besselj(0,k2*r1);
Hn=besselh(0,1,k2*r1);

Jd1n=1/rho1*k1*(besselj(-1,k1*r1)-besselj(1,k1*r1));
Jd2n=1/rho2*k2*(besselj(-1,k2*r1)-besselj(1,k2*r1));
Hdn=1/rho2*k2*(besselh(-1,1,k2*r1)-besselh(1,1,k2*r1));


a = (Jd2n/Jd1n-(J2n*Hdn)/(Hn*Jd1n))/(1-(J1n*Hdn)/(Hn*Jd1n));
b = a*J1n/Hn-J2n/Hn;

trs1 = a*besselj(0,k1*rr1);
trs2 = besselj(0,k2*rr2)+b*besselh(0,1,k2*rr2);

conve(1:2)=1;
m=1;

wait=waitbar(0,'Maximum computation time');

while conve(m)>tol
    
    waitbar(m/N)
  
    J1n=besselj(m,k1*r1);
    J2n=i^m*besselj(m,k2*r1);
    Hn=besselh(m,1,k2*r1);

    Jd1n=1/rho1*k1*(besselj(m-1,k1*r1)-besselj(m+1,k1*r1));
    Jd2n=i^m*1/rho2*k2*(besselj(m-1,k2*r1)-besselj(m+1,k2*r1));
    Hdn=1/rho2*k2*(besselh(m-1,1,k2*r1)-besselh(m+1,1,k2*r1));
	
     if isfinite((Jd2n/Jd1n-(J2n*Hdn)/(Hn*Jd1n))/(1-(J1n*Hdn)/(Hn*Jd1n)))==0 |...
	isfinite(J1n/Hn-J2n/Hn)==0
        break;
	disp('ZERO NOMINATOR IN FOURIER SERIES!!!');
	disp(['number of fourier modes=' num2str(m)]);
	disp(['truncation error=' num2str(conve(m))]);
    end

    a = (Jd2n/Jd1n-(J2n*Hdn)/(Hn*Jd1n))/(1-(J1n*Hdn)/(Hn*Jd1n));
    b = a*J1n/Hn-J2n/Hn;

  
  if m>=2
     trs_ex=trs;
  end 

  trr1 = a*besselj(m,k1*rr1);
  tth1 = 2*cos(m*th1);
  trs1 = trs1+trr1.*tth1;
 
  trr2 = i^m*besselj(m,k2*rr2)+b*besselh(m,1,k2*rr2);
  tth2 = 2*cos(m*th2);
  trs2 = trs2+trr2.*tth2;

  trs(ind1)=trs1;
  trs(ind2)=trs2;

  if m>=2
    conve(m+1)=max(abs(trs_ex-trs));
  end
  m=m+1;
  
  
end

disp(['number of fourier modes=' num2str(m)]);
disp(['truncation error=' num2str(conve(m))]);
close(wait)
