function ans = calculated(s)
ans = 0;
t = size(s) ;
t = t(1) ;
for i = 2 : t
    ans = ans +  sqrt(  (s(i,1) - s(i-1,1) )^2 +   (s(i,2) - s(i-1,2) )^2 ); 
end
end