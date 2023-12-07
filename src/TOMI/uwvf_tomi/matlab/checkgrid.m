function [h_min,h_max]=checkgrid(g,H)


N_H=length(H(:,1));
h_max=0;

for ii=1:N_H

   for jj=1:3
     
        if jj==3
            x1=g(H(ii,3),:);
            x2=g(H(ii,1),:);
        else
            x1=g(H(ii,jj),:);
            x2=g(H(ii,jj+1),:);
        end
  
        if ii==1 & jj==1
          h_min=norm(x2-x1);
          h_max=h_min;
        else
        h=norm(x2-x1);

            if h>h_max
               h_max=h;
            end  
            if h<h_min
               h_min=h;
            end

        end

    end

end 
      
               
          
        
