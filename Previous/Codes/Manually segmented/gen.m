%% make  contour initial

T = [23,29;
    23,131;
    121,131;
    121,29];
c = 1;
siz =2*(  (T(2,2) - T(1,2) ) +   (T(3,1) - T(3,2) ) + 2);

P  = zeros(siz,2);
for i = T(1,2) : T(2,2)
    P(c,1) = T(1,1) ;
    P(c,2) = i;
    c = c+1;
%     return;
end

for i = (T(1,1)+1 ): T(3,1)
    P(c,1) = i ;
    P(c,2) = T(3,2);
    c = c+1;
%     return;
end
for i =T(2,2)-1 :-1:  T(1,2)
    P(c,1) = T(3,1) ;
    P(c,2) = i;
    c = c+1;
    disp('ok');
%     return;
end

for i =   T(3,1) -1 : -1:T(1,1)
    P(c,1) = i ;
    P(c,2) = T(1,2);
%     return;
    c = c+1;
end
siz = size(P) ; 
for j = 1 : 1
    
    %    imshow(BW3{j});
    plot(P(:,2) ,P(:,1));
    hold on;
end