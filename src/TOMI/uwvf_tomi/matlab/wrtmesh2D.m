function wrtmesh2D(meshfile,g,H,NE,B)
%
% Write out a mesh file suitable for reading by F90
% wrtmesh(meshfile,g,H,B)
%
%
[fid message] = fopen(meshfile,'w');
ng=length(g);
nH=length(H);
NE(isnan(NE))=0;

if fid==-1
   disp('Unable to open file :-(')
   disp('Heres what the computer said:')
   disp(message)
   error('Fatal error, :-(')
else
   fprintf(fid,'%14i %14i \n',[ng,nH]);
   disp(['Mesh consists of ',num2str(nH),' triangles and '])
   disp(['                 ',num2str(ng),' vertices      '])
   for iH=1:nH
       fprintf(fid,'%14i %14i %14i %14i %14i %14i %14i %14i %14i %14i \n',[H(iH,:) ...
		    NE(iH,:) full(B(iH,:))]);
   end
   for ig=1:ng
       fprintf(fid,'%21.14e %21.14e \n',g(ig,:));
   end
end
fclose(fid)