close all;
clear all;

%load image. please input the full path to the image to 'imread' comma
image = imread('E:\DRIVE\test\images\01_test.tif');
figure, imshow(image)
%change data type to double and make it to [0 1] to calculate hsl
image=double(image)/255;
%convert to hsl color space
[h,s,L]=rgb2hsl(image);
figure, imshow(L)
%morphological closing
se = strel('disk',8);
gambarclose = imclose(L,se);
figure, imshow(gambarclose);
%median filter
halus=medfilt2(gambarclose,[20 20]);
figure, imshow(halus);
histogram = imhist(halus);
sumation = 0;
for(i=256:-1:1)
    if(histogram(i)>0)
        summation = sumation+histogram(i); 
        if(summation>600)
            threshold = i/256;
            break;
        end
    end
end

figure, imhist(halus);
average = mean2(halus);
for(i=1:size(halus,1))
    for(j=1:size(halus,2))
        if(halus(i,j)>threshold)
            bw(i,j) = 1;
        else
            bw(i,j) = 0;
        end
        if(halus(i,j) < 0.1)
            halus(i,j) = average;
        end
    end
end
figure, imshow(bw);
[a b] = max(halus(:));
[y x] = ind2sub(size(halus),b);
hold on
%location of the brightest pixel on image
%used for getting initial point for active contour
viscircles([x y], 10,'EdgeColor','b');
hold off
P = [y+30 x; y x+30; y-30 x; y x-30];
%option for active contour
Options=struct;
Options.Verbose=false;
Options.Iterations=300;
Options.Wedge=3;
[O,J]=Snake2D(halus,P,Options);
boundaries = bwboundaries(bw);
figure, imshow(image);
hold on
%plot result on original image
for k = 1:length(boundaries)
   B = boundaries{k};
   plot(B(:,2), B(:,1), 'w', 'LineWidth', 1)
end
for k = 1:length(O)
   plot(O(:,2), O(:,1), 'b', 'LineWidth', 1)
end
%segment the result
bw(:,:,2) = bw;
bw(:,:,3) = bw(:,:,1);
resultL = image;
resultL(bw == 0) = 0;
figure, imshow(resultL);
J(:,:,2) = J;
J(:,:,3) = J(:,:,1);
resultactive = image;
resultactive(J ==0) = 0;
figure, imshow(resultactive);