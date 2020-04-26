imshow(closeBW);
s2  = regionprops(closeBW,'centroid','MajorAxisLength','MinorAxisLength');
centroids = cat(1, s2.Centroid);
imshow(closeBW)
hold on
plot(centroids(:,1), centroids(:,2), 'b*')
hold off

centers = s2.Centroid;
diameters = mean([s2.MajorAxisLength s2.MinorAxisLength],2);
radii = diameters/2;
P = circle(centroids(:,1), centroids(:,2),3*radii);