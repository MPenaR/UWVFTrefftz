function plcigrid(g,H);
% The following variables are needed: 
% g = grid point matrix, 
% H = index matrix of points of g 
% If FILL is defined the triangles are plotted filled.
% If Int is defined, the intrior points are also marked

% J. Kaipio, 11.4.1994. Interior points added 18.6.1994

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

for ii=1:nH
  Hii=g(H(ii,:),:);
  Hii=[Hii;Hii(1,:)];
  if exist('FILL')
    hHii=fill(Hii(:,1),Hii(:,2),[1 1 1]);
  else
    hHii=plot(Hii(:,1),Hii(:,2));
  end
end
hold off



