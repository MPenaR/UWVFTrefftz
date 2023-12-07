
clear

% define and draw two circles

circle1=circ2(0,0,0.10);
circle2=circ2(0,0,0.05);

fem.geom=circle1-circle2;
figure(1),geomplot(fem.geom,'edgelabels','on','sublabels','on'),axis equal

hmax = [26];
Nh   = length(hmax);

for ih = 1:Nh
  
  if hmax(ih)<10
    z0 = '0';
  else
    z0 =  '';
  end
  
  % generate mesh
  fem.mesh=meshinit(fem.geom,'hmax',hmax(ih)*1e-3);       
  figure(2),meshplot(fem),axis equal

  sdind = 1;
  elabels = [1 2; 2 2; 3 3; 4 3; 5 2; 6 3; 7 3; 8 2];
  
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
  save(['../data/mesh_sc' z0 int2str(hmax(ih)) '.mat'],...
       'fem','g','gp','H','NE','NOR','B','h');
  
  % write .dat-file
  wrtmesh2D(['../data/mesh_sc' z0 int2str(hmax(ih)) '.dat'],g,H,NE,B)

end
