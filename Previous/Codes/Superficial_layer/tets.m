closeBW = Store;
for i = 1 : 15 
    closeBW = imerode(closeBW,1);
    imshow(closeBW);
end