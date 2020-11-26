clc;
close all;
jpegFiles = dir('1');
numfiles = length(jpegFiles);
op=1;
for i =3 : numfiles
    
filename = jpegFiles(i).name;
% reading and displaying the image
I = imread(filename);
% figure,imshow(I);
 [a ,b,c] = size(I);
  I=I(ceil(0.10*a):ceil(0.90*a),ceil(0.10*b):ceil(0.90*b),1:3); 

grayImg=rgb2gray(I);
figure
h(1)=imshow(grayImg);

%% 

eyeMask= im2bw(grayImg,graythresh(grayImg));
eyeMask= imfill(eyeMask,'holes');
eyeMask=bwareaopen(eyeMask,100);
% imshow(eyeMask);

%apply mask to eleminate background noise
grayImg(~eyeMask)=0;
set(h(1),'cdata',grayImg);


% Segment the vessels
vesselmask=edge(grayImg,'canny',0.10,1);
% imshow(vesselmask);
vs=imgca;


% dilation
vesselmask=imdilate(vesselmask,strel('disk',7));
%figure, imshow(vesselmask);


vesselmask=bwmorph(vesselmask,'skel',Inf);
vesselmask=bwmorph(vesselmask,'spur',5);
%figure, imshow(vesselmask)


branchPoints=bwmorph(vesselmask,'branch',1);
branchPoints=imdilate(branchPoints,strel('disk',2));
% figure, imshow(branchPoints)
bp=imgca;

% vesselmask = bwmorph(vesselmask,'dilate');
vesselmask=vesselmask & ~ branchPoints;
% figure, imshow(vesselmask)
% linkaxes([vs,bp],'xy')


% % %then we discard short segments
minLength=25;
vesselSegs=bwareaopen(vesselmask,minLength,8);
%figure, imshow(vesselSegs)
K = vesselSegs ; 

stats = regionprops('table',K,'Area', 'PixelList');
o = size(stats) ;
%figure(8)
c =1;
for i = 1  : o(1) 
    t = size(stats.PixelList{i} ) ;
    
    t = t(1) ;
    s = stats.PixelList{i} ;
%     plot(s);
%     hold on;
    dist =sqrt   ( ( s(1,1) - s(t,1) )^2  + ( s(1,2) - s(t,2) )^2 )  ; 
    if calculated(s) / dist   < 5 
        T(c) = calculated(s) / dist ;
        c = c+1;
    end
    
    
end
L(op) = mean(T) ; 
op = op+1;
% BW3 = bwmorph(K,'skel');

end
%%%%%%%%  ---------------now measure the tourtousity of these vessels
%% 

