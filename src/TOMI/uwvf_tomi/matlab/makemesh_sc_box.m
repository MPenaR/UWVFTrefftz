
clear

% define and draw two circles

circle1=rect2(-2,2,-2,2);
circle2=circ2(0,0,1);

fem.geom=circle1-circle2;
figure(1),geomplot(fem.geom,'edgelabels','on','sublabels','on'),axis equal

hmax = [0.5];
Nh   = length(hmax);

for ih = 1:Nh
  
 
  % generate mesh
  fem.mesh=meshinit(fem.geom,'hmax',hmax(ih));       
  figure(2),meshplot(fem),axis equal

  sdind = 1;
  elabels = [1 4; 2 4; 3 4; 4 4; 5 3; 6 3; 7 3; 8 3];
  
  % transform to the UWVF geometry
  [g,H,NE,NOR,B] = femlab2uwvf2d(fem,sdind,[],elabels);
  [minh,maxh]    = checkgrid(g,H);
  h              = [minh maxh];
  maxh 
  
  % nodes in  polar coordinates
  [th,r] = cart2pol(g(:,1),g(:,2));
  ind = find(th<0);
  th(ind) = th(ind)+2*pi;
  gp = [r th];
  
  % save
  save(['../data/mesh_sc_box'  num2str(hmax(ih)) '.mat'],...
       'fem','g','gp','H','NE','NOR','B','h');
  
  % write .dat-file
  wrtmesh2D(['../data/mesh_sc_box' num2str(hmax(ih)) '.dat'],g,H,NE,B)

end
