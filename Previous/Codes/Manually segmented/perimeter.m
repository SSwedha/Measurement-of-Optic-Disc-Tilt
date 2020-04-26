function Perimeter = perimeter(P,k,sd,t)
sz = size(P);
sum = 0;
for i =1 : sz(1) - 1
   sum = sum + sqrt((P(i,1) - P(i+1,1))^2 + (P(i,2) - P(i+1,2))^2 ); 
end

Perimeter = sum * sd(t)/k ; 

end