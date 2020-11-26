function closeBW = removeb(closeBW) 
im = closeBW ; 
closeBW2 = closeBW ;
for t = 1 : 2
    closeBW2 = closeBW ;

%    closeBW = bwareaopen(closeBW,max(stats.Area)-1);
%         bound = bwboundaries(closeBW);
%         s = size(bound);
%         for j = 1 : s(1)
%        boundnew=bound{j};
%         end
%         disp(closeBW(boundnew(1,:))) ; 
%         
%         closeBW(boundnew) = 0;
%          disp(closeBW(boundnew(1,:))) ; 
% end

sd =  size(closeBW) ;
for i = 2 : sd(1) - 1
    for j = 2 : sd(2) - 1
        
        if ( closeBW2(i-1,j) ==  0 ||closeBW2(i+1,j) == 0 ||   closeBW2(i,j-1) == 0 ||closeBW2(i,j+1) ==0) 
            closeBW(i,j) =  0  ; 
        end
    end
end



end

% figure(3) 
%         imshow([im,closeBW]); 
%         hold on;
%         input('');

end