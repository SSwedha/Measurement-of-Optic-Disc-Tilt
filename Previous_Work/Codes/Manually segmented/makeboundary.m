function K = makeboundary(closeBW,P,K) 
     sizo = size(closeBW) ; 
     new_BW= zeros(sizo) ;
     sizet = size(P);
     xq = zeros(sizo(1)*sizo(2),1);
     yq=  zeros(sizo(1)*sizo(2),1);
     c= 0;
     for i = 1 : sizo(1)
         for j = 1 : sizo(2)
             c = c+1;
             xq(c) = i;
             yq(c) = j;
         end
     end
     in = inpolygon(xq,yq,P(:,1),P(:,2));
%      figure

% plot(P(:,1),P(:,2)) % polygon
% axis equal
% 
% hold on
% plot(xq(in),yq(in),'r+') % points inside
% plot(xq(~in),yq(~in),'bo') % points outside
% hold off

tu = [xq(in) , yq(in) ] ;

sd = size(tu);
for i = 1 : sd
    a = tu(i,:);
    au = ceil(a(1)+0.32*406);
    ad =ceil( a(2)+0.32*412);
    K(au,ad,2) = 255;
    
    
end
 
    
     
     
end