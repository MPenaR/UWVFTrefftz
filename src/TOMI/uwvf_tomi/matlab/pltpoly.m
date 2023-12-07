function pltpoly(g,H,val);
% The following variables are needed: 
% g = grid point matrix, 
% H = index matrix of points of g 
% val = value to be plotted
t=[0:.1:2.1*pi]';
nH=max(size(H));
x_min=min(g(:,1));
x_max=max(g(:,1));
y_min=min(g(:,2));
y_max=max(g(:,2));
%plot(r_max*sin(t),r_max*cos(t),'g')
axis([x_min x_max y_min y_max])
axis(axis)
axis('square')
hold on
plot(g(:,1),g(:,2),'w.');
%if exist('Int'), plot(g(Int,1),g(Int,2),'wo'); end
colormap(jet)
for ii=1:nH
  Hii=g(H(ii,1:3),:);
  Hii=[Hii;Hii(1,:)];
  if val(ii)==1
    hHii=fill(Hii(:,1),Hii(:,2),[1,0,0]);
  else
    hHii=fill(Hii(:,1),Hii(:,2),[0,1,0]);
  end
end
hold off



