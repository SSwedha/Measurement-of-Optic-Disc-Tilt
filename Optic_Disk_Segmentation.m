% This is a program for segmenting the optic disc and cup boundaries from a %fundus image. Written for %calculating the HCDR and VCDR for diagnosing the glaucoma
% Authors : Ahmed Almazroa, 2016
% email: myh_300@hotmail.com
% Copyright (c) 2014-2016 by Ahmed Almazroa
% U can use an entire fundus image in format of TIFF or JPG
% input - give the image file name as input. eg :- image2prime
clc;
clear all;
close all;
tic
%% Main Input
thLevel =3;% starting with threshold number three
Eye_side = 'R';%can be Left or Right
im_num = 1; %image number

max_pixl_yc = 10; %maximum distance between the disc and cetroid on Yaxis

maxArea = 18000; %maximum cup area
minArea = 2000; %minimum cup area

Extra_side = 0.45;% percentage of extra radius

Radius_up = 0.2;% percentage of extra radius
Radius_down = 0.2;% percentage of extra radius
Radius_right = 0.0;% percentage of extra radius
Radius_left = 0.0;% percentage of extra radius

Level2Set = 1; %if 1 then Level2Set is used to detect the disc (JPG) NONmydriatric retinal camera else (0) SmoothMuch will be used

% Do not chnage this data
error = 500;
lower_area = 1000;

% Blood Vessel Extraction
L = 45;
Extra_radius = [Radius_up,Radius_down,Radius_right,Radius_left];
Enhanced = 1; %Enable enhance at the beginning
Error = 0; %There is no error

%% Change the input images number and name prefix.
%I=['image',num2str(im_num),'.tif'];
%I=['image',num2str(im_num),'prime.tif'];
%I=['image',num2str(im_num),'.jpg'];
%I=['image',num2str(im_num),'prime.jpg'];
%I = '_OD.jpeg';
I = 'download.jpg';
fname = I;
fprintf('\nWorking on %s\n',I)
I_temp = ['Localized',fname];
inImg_temp1 = imread(I);

inImg_temp2 = localization(inImg_temp1);
imwrite(inImg_temp2,I_temp)
fprintf('\nLocalization is complete and imgae is saved. Now working on threshold and enhancement \n')
inImg = imread(I_temp);
[M,N,~] = size(inImg);
inImg1 = inImg;
clear inImg_temp1
clear inImg_temp2

% Start cup extraction with 3 level thresholding to calculate range of area
radiusError = 1;

while radiusError ==1
    % Generate 3 level thresholding for estimated cup size
    [~, ThresImg_temp] = Thresholding(inImg,3);
    [M,N,~] = size(inImg);
    Img_temp1 = zeros([M,N]);
    Thres_temp = unique(ThresImg_temp);
    n = length(Thres_temp)+1;
    mean_area = 0;
    while mean_area<lower_area
        n = n-1;
        if n==0
            mean_area = lower_area;
        else
            Img_temp1(ThresImg_temp>=Thres_temp(n)) = 1;
            mean_area = sum(nonzeros(Img_temp1));
        end
    end
    
    % Estimate the range of cup size
    max_area = mean_area + error;
    min_area = mean_area - error;

    % Perform main thresholding to detect the cup region
    if Enhanced==1
        [~, thresImg] = Thresholding(inImg,thLevel);
    else
        [~, thresImg] = ThresholdingNE(inImg,thLevel);
    end
    
    %thresImg = ThresImg_temp;
    Threshold = unique(thresImg);
    Threshold = Threshold';

    % Iterate until the calcualted area falles between the range
    area = 0;
    for i = length(Threshold):-1:1
        area = area + sum(sum(thresImg==Threshold(i)));
        if area>=min_area
            if area>max_area
                n = i+1;
            else
                n = i;
            end
            break;
        end
    end
    
    n = min(n,length(Threshold));
    Img_temp2 = zeros(M,N);
    
    for i=1:1:M
        for j=1:1:N
            if thresImg(i,j)>= Threshold(n)
                Img_temp2(i,j) = 1;
            end
        end
    end
    
    %% Find tentative circle that covers the cup region
    
    count = 0;
    r0 = 1;
    try
        while count==0
            [c,rr] = PaintCircle(Img_temp2,r0);
            [count,~] = size(c);
            % Repeat if centre is not found by increasing r0
            if count<=0
                r0=r0+1;
            end
        end
        
        %% Based on the circle and eye side, find the ellipse that fits the largest number of white pixels
        areaa =[];

        for i=1:1:length(c(:,1))
            % Find the masking filled circle
            [Img_ellipse,~,~,~,~,~] = FillEllipse(c(i,1),c(i,2),rr(i),Extra_radius,M,N,Eye_side,Extra_side);
            ImgCup = Img_ellipse.*Img_temp2;
            areaa = [areaa,nnz(ImgCup)];
        end
        
        [~,index] = max(areaa);
        cx = round(c(index,1));
        cy = round(c(index,2));
        r = round(rr(index));
        [Img_ellipse,x_ellipse,y_ellipse,a,b,x_set] = FillEllipse(cx,cy,r,Extra_radius,M,N,Eye_side,Extra_side);
        AreaEllipse = nnz(Img_ellipse);

        if AreaEllipse<minArea || AreaEllipse>maxArea
            fprintf('\n Radius out of range. Adjusting threshold and enhancing\n')
            if thLevel == 3 && Enhanced==1
                thLevel = 2;
            elseif thLevel==2
                thLevel = 4;
                Enhanced = 0;
            elseif thLevel==4
                thLevel = 3;
                Enhanced = 0;
            else
                Error = 1;
                radiusError =0;
            end    
        else            
            radiusError =0;
        end
        
    catch
        fprintf('\n An error occured. Adjusting threshold and enhancing\n')
        if thLevel == 3 && Enhanced==1
            thLevel = 2;
        elseif thLevel==2
            thLevel = 4;
            Enhanced = 0;
        elseif thLevel==4
            thLevel = 3;
            Enhanced = 0;
        else
            Error = 1;
            radiusError =0;
        end    
    end%try
    
end%radiusError ==1;

if Error ==0
    %% Print Threshold leven and Enhanced
    fprintf('\nThreshold level = %d',thLevel)
    if Enhanced==1
        fprintf('\nEnhancement is enabled')
    else
        fprintf('\nEnhancement is disabled')
    end
    
    %% Extract the blood vessels
    % Extract blood vessels based on green channel
    if Level2Set ==1
        [filledDisc,edgeDisc,inImg,bloodVessel1,y_disc,x_disc] = discExtractDoubleLevelSet(inImg1);
    else
        [filledDisc,edgeDisc,inImg,bloodVessel1,y_disc,x_disc] = discExtractSmoothMuch(inImg1);
    end
    
    for i=1:1:M
        for j = 1:1:N
            % paint disc border
            if edgeDisc(i,j)>=1 %&& i>1 && j>1
                inImg(i,j,:)=[0 255 255];
                inImg(i-1,j,:)=[0 255 255];
                inImg(i,j-1,:)=[0 255 255];
                inImg(i-1,j-1,:)=[0 255 255];
            end    
        end
    end
    
    areaDisc = nnz(filledDisc);
    bloodVessel = bloodVessel1.*filledDisc;
    bloodVessel2 = bloodVessel;
    
    %% Calculate and adjust/set variables
    xc_cup(1) = x_ellipse;
    if abs(y_disc-y_ellipse)>max_pixl_yc
        fprintf('\nY-coordinate is adjusted\n')
        if y_ellipse>y_disc
            y_ellipse = y_disc + 10;
        else
            y_ellipse = y_disc - 10;
        end
    end
    
    if Eye_side == 'R'
        mul = 1;
        xc_edge = max(x_set);
    else
        mul = -1;
        xc_edge = min(x_set);
    end
    
    %% Find BloodVessel
    [Edges,X,Img_square,max_pixl] = FindBloodVesselPoint2(M,N,bloodVessel,L,cx,cy,Eye_side);

    x_square = cx + mul * round((X-1)*L+L/2);
    bloodVessel1 = bloodVessel.*Img_square;
    border1 = inImg; % This is to show the square box and edge of the ellipse
    edges = edge(Img_ellipse,'sobel');
    for i=1:1:M
        for j = 1:1:N
            % paint ellipse border
            if edges(i,j)>=1
                border1(i,j,:)=[0 0 255];
                border1(i-1,j,:)=[0 0 255];
                border1(i,j-1,:)=[0 0 255];
                border1(i-1,j-1,:)=[0 0 255];
            end
        end
    end
    
    border1(y_ellipse,xc_cup(1),:) = [255,255,255];
    figure;imshow(border1);title('No adjustment');
    
    % II = ['EllipseSquare_Cup_NoAdjust', fname];
    % imwrite(border1,II)
    
    %Shift to the near edge
    dist_E2NE = round(cx + mul*(X-1)*L)-xc_edge;
    %Shift to the medium edge
    dist_E2M = round(x_square-xc_edge);
    %Shift to the far edge
    dist_E2FE = round(cx + mul*X*L)-xc_edge;
    
    Img_ellipseM = MakeFilledEllipse(M,N,x_ellipse,y_ellipse,a,b,dist_E2M ,0);
    Img_ellipseNE = MakeFilledEllipse(M,N,x_ellipse,y_ellipse,a,b,dist_E2NE,0);
    Img_ellipseFE = MakeFilledEllipse(M,N,x_ellipse,y_ellipse,a,b,dist_E2FE,0);
    
    xc_cup(2:4) = [xc_cup(1) + dist_E2NE, xc_cup(1) + dist_E2M, xc_cup(1) + dist_E2FE];
    
    overlapped = [sum(sum(Img_ellipse.*edgeDisc)),sum(sum(Img_ellipseNE.*edgeDisc)),sum(sum(Img_ellipseM.*edgeDisc)),sum(sum(Img_ellipseFE.*edgeDisc))];
    
    % below if for the diplay propose
    edgesNE = edge(Img_ellipseNE,'sobel');
    borderNE = inImg;
    
    for i=1:1:M
        for j = 1:1:N
            if edgesNE(i,j)>=1
                borderNE(i,j,:)=[0 0 255];
                borderNE(i-1,j,:)=[0 0 255];
                borderNE(i,j-1,:)=[0 0 255];
                borderNE(i-1,j-1,:)=[0 0 255];
            end
        end
    end
    
    borderNE(y_ellipse,xc_cup(2),:) = [255,255,255];
    figure;imshow(borderNE);title('Near-edge Adjustment');
    
    % below if for the diplay propose
    edgesM = edge(Img_ellipseM,'sobel');
    borderM = inImg;
    
    for i=1:1:M
        for j = 1:1:N
            if edgesM(i,j)>=1
                borderM(i,j,:)=[0 0 255];
                borderM(i-1,j,:)=[0 0 255];
                borderM(i,j-1,:)=[0 0 255];
                borderM(i-1,j-1,:)=[0 0 255];
            end
        end
    end
    
    borderM(y_ellipse,xc_cup(3),:) = [255,255,255];
    figure;imshow(borderM);title('Mid-Point Adjustment');
    
    % below if for the display propose
    edgesFE = edge(Img_ellipseFE,'sobel');
    borderFE = inImg;
    
    for i=1:1:M
        for j = 1:1:N
            if edgesFE(i,j)>=1
                borderFE(i,j,:)=[0 0 255];
                borderFE(i-1,j,:)=[0 0 255];
                borderFE(i,j-1,:)=[0 0 255];
                borderFE(i-1,j-1,:)=[0 0 255];
            end
        end
    end
    borderFE(y_ellipse,xc_cup(4),:) = [255,255,255];
    figure;imshow(borderFE);title('Far-edge Adjustment');

    % II = ['EllipseSquare_Cup_FarEdgeAdjust_', fname];
    % imwrite(borderFE,II)
   
    diff_x = abs(x_disc-xc_cup);
    cutOff = find(overlapped>0);
    diff_x(cutOff) = inf;
    [~,ind] = min(diff_x);
    
    % Display Disc statics
    areaDisc
    x_disc
    y_disc
    
    [~,~,~,~,vDist_disc,hDist_disc] = maxminPoints(filledDisc);
    
    II = ['EllipseSquare_Cup_BestAdjust_', fname];

    if ind==1
        fprintf('\n No adjustment is needed for x-coordinate\n')
        areaCUP = nnz(Img_ellipse);
        imwrite(border1,II);
        [~,~,~,~,vDist_cup,hDist_cup] = maxminPoints(Img_ellipse);
    elseif ind==2
        fprintf('\n Near-edge adjustment is the best for x-coordinate\n')
        areaCUP = nnz(Img_ellipseNE);
        imwrite(borderNE,II);
        [~,~,~,~,vDist_cup,hDist_cup] = maxminPoints(Img_ellipseNE);
    elseif ind==3
        fprintf('\n Adjustment to the middle is the best for x-coordinate\n')
        areaCUP = nnz(Img_ellipseM);
        imwrite(borderM,II);
        [~,~,~,~,vDist_cup,hDist_cup] = maxminPoints(Img_ellipseM);
    else
        fprintf('\n Far-edge adjustment is the best for x-coordinate\n')
        areaCUP = nnz(Img_ellipseFE);
        imwrite(borderFE,II);
        [~,~,~,~,vDist_cup,hDist_cup] = maxminPoints(Img_ellipseFE);
    end
    
    areaCUP
    y_cup = y_ellipse;
    x_cup = xc_cup(ind);
    %vDist_cup
    %hDist_cup
    ratioVerticalDistance = vDist_cup/vDist_disc;
    ratioHorizontalDistance = hDist_cup/hDist_disc;

else%Error ==0
    fprintf('\n An error occured. This image cannot be processed for cup detection')

end%Error ==0

toc
h = msgbox({sprintf('Summary for %s',fname) ...

sprintf('\nDisc:') ...
sprintf('Area of the disc:\t %d',areaDisc) ...
sprintf('x-axis of the disc:\t %d',x_disc) ...
sprintf('y-axis of the disc:\t %d',y_disc) ...
sprintf('\nCup:') ...
sprintf('Area of the cup:\t %d',areaCUP) ...
sprintf('x-axis of the cup:\t %d',x_cup) ...
sprintf('y-axis of the cup:\t %d',y_cup) ...
sprintf('\nCup to disc ratio:') ...
sprintf('Verticle distance:\t %.4f',ratioVerticalDistance) ...
sprintf('Horizontal distance:\t %.4f',ratioHorizontalDistance)},'Results');
%%

function [bw] = better_bloodVesselExtract(gray)
    %input parameter: rgb is a input_imgage;
    %output parameter: bw_bloodVessel is a binary image.1 means blood vessel pixel
    %Example:
    % rgb=imread('123.tif');
    % [bw_bloodVessel]=bloodVesselExtract(gray_img);
    % I recommendend the green channel of the RGB.
    %black top_hat transformation
    se= strel('disk',20);
    no_blood_vessel = imclose(gray,se);
    diff =no_blood_vessel-gray;% diff contains the information of the blood vessel.x
    %thresholding segmentation
    th=graythresh(diff);
    bw=im2bw(diff,th);
    %remove the small spot
    [L,num] = bwlabel(bw);
    for i=1:num
        [x,y]=find(L==i);
        area=size(x);
        if area<10
            bw(x,y)=0;
        end
    end
end

function [C1,C2] = binaryfit(phi,U,epsilon)
    % [C1,C2]= binaryfit(phi,U,epsilon) computes c1 c2 for optimal binary fitting
    % input:
    % U: input image
    % phi: level set function
    % epsilon: parameter for computing smooth Heaviside and dirac function
    % output:
    % C1: a constant to fit the image U in the region phi>0
    % C2: a constant to fit the image U in the region phi<0
    %
    % created on 04/26/2004
    % author: Chunming Li
    % email: li_chunming@hotmail.com
    % Copyright (c) 2004-2006 by Chunming Li
    H = Heaviside(phi,epsilon); %compute the Heaveside function values
    a= H.*U;
    numer_1=sum(a(:));
    denom_1=sum(H(:));
    C1 = numer_1/denom_1;
    b=(1-H).*U;
    numer_2=sum(b(:));
    c=1-H;
    denom_2=sum(c(:));
    C2 = numer_2/denom_2;
end

function [bdy,bdx] = backward_gradient(f)
    % function [bdx,bdy]=backward_gradient(f);
    %
    % created on 04/26/2004
    % author: Chunming Li
    % email: li_chunming@hotmail.com
    % Copyright (c) 2004-2006 by Chunming Li
    [nr,nc]=size(f);
    bdx=zeros(nr,nc);
    bdy=zeros(nr,nc);
    bdx(2:nr,:)=f(2:nr,:)-f(1:nr-1,:);
    bdy(:,2:nc)=f(:,2:nc)-f(:,1:nc-1);
end

function B = BoundMirrorEnsure(A)
    [m,n] = size(A);
    if (m<3 | n<3)
        error('either the number of rows or columns is smaller than 3');
    end
    yi = 2:m-1;
    xi = 2:n-1;
    B = A;
    B([1 m],[1 n]) = B([3 m-2],[3 n-2]); % mirror corners
    B([1 m],xi) = B([3 m-2],xi); % mirror left and right boundary
    B(yi,[1 n]) = B(yi,[3 n-2]); % mirror top and bottom boundary
end

function B = BoundMirrorExpand(A)
    [m,n] = size(A);
    yi = 2:m+1;
    xi = 2:n+1;
    B = zeros(m+2,n+2);
    B(yi,xi) = A;
    B([1 m+2],[1 n+2]) = B([3 m],[3 n]); % mirror corners
    B([1 m+2],xi) = B([3 m],xi); % mirror left and right boundary
    B(yi,[1 n+2]) = B(yi,[3 n]); % mirror top and bottom boundary
end

function B = BoundMirrorShrink(A)
    [m,n] = size(A);
    yi = 2:m-1;
    xi = 2:n-1;
    B = A(yi,xi);
end

function onlyDisc = cropDisc(inImg,refMatrix)
    [M,N] = size(refMatrix);
    if ndims(inImg)>2
        onlyDisc = inImg;
        for i =1:1:M
            for j = 1:1:N
                if refMatrix(i,j)<=0
                    onlyDisc(i,j,:) = [0,0,0];
                end
            end
        end    
    else
        onlyDisc = zeros(M,N);
        for i =1:1:M
            for j = 1:1:N
                if refMatrix(i,j)>0
                    onlyDisc(i,j) = inImg(i,j);
                end
            end
        end
    end
end

function K = curvature(f)
    % K=curvature(f);
    % K=div(Df/|Df|)
    % =(fxx*fy^2+fyy*fx^2-2*fx*fy*fxy)/(fx^2+fy^2)^(3/2)
    % created on 04/26/2004
    % author: Chunming Li
    % email: li_chunming@hotmail.com
    % Copyright (c) 2004-2006 by Chunming Li
    [f_fx,f_fy]=forward_gradient(f);
    [f_bx,f_by]=backward_gradient(f);
    mag1=sqrt(f_fx.^2+f_fy.^2+1e-10);
    n1x=f_fx./mag1;
    n1y=f_fy./mag1;
    mag2=sqrt(f_bx.^2+f_fy.^2+1e-10);
    n2x=f_bx./mag2;
    n2y=f_fy./mag2;
    mag3=sqrt(f_fx.^2+f_by.^2+1e-10);
    n3x=f_fx./mag3;
    n3y=f_by./mag3;
    mag4=sqrt(f_bx.^2+f_by.^2+1e-10);
    n4x=f_bx./mag4;
    n4y=f_by./mag4;
    nx=n1x+n2x+n3x+n4x;
    ny=n1y+n2y+n3y+n4y;
    magn=sqrt(nx.^2+ny.^2);
    nx=nx./(magn+1e-10);
    ny=ny./(magn+1e-10);
    [nxx,nxy]=gradient(nx);
    [nyx,nyy]=gradient(ny);
    K=nxx+nyy;
end

function Delta_h = Delta(phi, epsilon)
    % Delta(phi, epsilon) compute the smooth Dirac function
    %
    % created on 04/26/2004
    % author: Chunming Li
    % email: li_chunming@hotmail.com
    % Copyright (c) 2004-2006 by Chunming Li
    Delta_h=(epsilon/pi)./(epsilon^2+ phi.^2);
end

function [phi,edge_disc2,Img,bloodVessel,xc,yc] = discExtractDoubleLevelSet(inImg)
    U=inImg(:,:,1);
    G=inImg(:,:,2);
    mask=better_bloodVesselExtract(G);
    bloodVessel = mask;
    U(find(U<30))=100;
    hsvImg=rgb2hsv(inImg);
    V=hsvImg(:,:,3);
    V=grayStretch(V);
    V=FastInpaint(V,mask,500);
    U=FastInpaint(U,mask,500);
    G=FastInpaint(G,mask,500);
    % get the size
    [nrow,ncol] =size(U);
    ic=nrow/2;
    jc=ncol/2;
    r=90;
    phi_0 = sdf2circle(nrow,ncol,ic,jc,r);
    delta_t = 0.1;
    lambda_1=1;
    lambda_2=1;
    nu=0;
    h = 1;
    epsilon=8;
    mu = 0.01*255*255;
    I=U;
    % iteration should begin from here
    phi=phi_0;
    numIter = 10;
    for k=1:90
        phi=evolution_cv(I, phi, mu, nu, lambda_1, lambda_2, delta_t, epsilon, numIter); % update level set function
    end
    %%
    phi=im2bw(phi,0);
    phi=bwareaopen(phi,500);
    phi=imcomplement(phi);
    phi=bwareaopen(phi,500);
    edge_disc=edge(phi);
    se=strel('disk',1);
    edge_disc=imclose(edge_disc,se);
    %%
    [m,n]=size(edge_disc);
    cc = bwconncomp(edge_disc);
    numFields = getfield(cc,'NumObjects');
    
    if(numFields > 1)
        S = regionprops(cc, 'Area');
        P = max([S.Area]);
        L = labelmatrix(cc);
        temp = ismember(L, find([S.Area] >= P));
        for i = 1:m
            for j = 1:n
                if(temp(i,j) == 0)
                    edge_disc(i,j) = 0;
                end
            end
        end
    end
    
    %%
    [x,y]=find(edge_disc>0);
    U2=U(min(x):max(x),min(y):max(y));
    %%
    [nrow,ncol] =size(U2);
    ic=round(nrow/2);
    jc=round(ncol/2);
    r=90;
    phi_0 = sdf2circle(nrow,ncol,ic,jc,r);
    %%
    edge_disc2=zeros(size(edge_disc));
    %left
    Uleft=U2(:,1:jc+10);
    phi_0left=phi_0(:,1:jc+10);
    % I=['C:\Users\weiwei\Desktop\X Y for the localized\Disc\Ibn1\left',num2str(imgNum),'.tif'];
    % U3=uint8(Uleft);
    % imwrite(U3,I);
    I=Uleft;
    phi=phi_0left;
    
    for k=1:90
        phi=evolution_cv(I, phi, mu, nu, lambda_1, lambda_2, delta_t, epsilon, numIter); % update level set function
    end
    
    phi=im2bw(phi,0);
    phi=bwareaopen(phi,100);
    phi=imcomplement(phi);
    phi=bwareaopen(phi,200);
    edge_templeft=edge(phi);
    edge_disc2(min(x):max(x),min(y):min(y)+jc+9)=edge_templeft;
    %%
    %right
    Uright=U2(:,jc-10:end);
    phi_0right=phi_0(:,jc-10:end);
    % I=['C:\Users\weiwei\Desktop\X Y for the localized\Disc\Ibn1\right',num2str(imgNum),'.tif'];
    % U3=uint8(Uright);
    % imwrite(U3,I);
    I=Uright;
    phi=phi_0right;
    
    for k=1:90
        phi=evolution_cv(I, phi, mu, nu, lambda_1, lambda_2, delta_t, epsilon, numIter); % update level set function
    end
    
    phi=im2bw(phi,0);
    phi=bwareaopen(phi,100);
    phi=imcomplement(phi);
    phi=bwareaopen(phi,200);
    edge_tempright=edge(phi);
    edge_disc2(min(x):max(x),min(y)+jc-11:max(y))=edge_tempright;
    %%
    %%
    se=strel('disk',10);
    edge_disc2=imclose(edge_disc2,se);
    [m,n]=size(edge_disc2);
    cc = bwconncomp(edge_disc2);
    numFields = getfield(cc,'NumObjects');
    
    if(numFields > 1)
        S = regionprops(cc, 'Area');
        P = max([S.Area]);
        L = labelmatrix(cc);
        temp = ismember(L, find([S.Area] >= P));
        for i = 1:m
            for j = 1:n
                if(temp(i,j) == 0)
                    edge_disc2(i,j) = 0;
                end
            end
        end
    end
    
    dif=4;
    se=strel('disk',1);
    db=0;
    recordedx=find(edge_disc2>0);
    
    while(dif>3)
        [edge_disc2,dif]=eliminateBlank(edge_disc2);
        edge_disc2=imclose(edge_disc2,se);
        db=db+1;
        if(db==10)
            break;
        end
    end
    recordedx1=find(edge_disc2>0);
    edge_disc2=imdilate(edge_disc2,se);
    [x3]=find(edge_disc2);
    edge_disc2=edgeOptimize(edge_disc2,10);
    se=strel('disk',5);
    edge_disc2=imclose(edge_disc2,se);
    se=strel('disk',2);
    edge_disc2=imdilate(edge_disc2,se);
    phi=eTp(edge_disc2);
    %phi is the final mask of the disc
    %figure, imshow(phi)
    edge_disc2=edge(phi);
    % edge of the final disc image
    %figure, imshow(edge_disc2)
    [x,y]=find(edge_disc2>0);
    xc=round((min(x)+max(x))/2);
    yc=round((min(y)+max(y))/2);
    num=size(find(phi==1),1);
    
    Img=inImg;
    Img1=inImg;
    Img2=inImg;
    [x]=find(edge_disc>0);
    [x2]=find(edge_disc2>0);
    r=Img(:,:,1);
    g=Img(:,:,2);
    b=Img(:,:,3);
    % r(x)=0;
    % g(x)=255;
    % b(x)=0;
    
    r(x2)=0;
    g(x2)=0;
    b(x2)=255;
    % r(x3)=0;
    % g(x3)=0;
    % b(x3)=0;
    Img(:,:,1)=r;
    Img(:,:,2)=g;
    Img(:,:,3)=b;
    %figure, imshow(Img)
end

function [phi,edge_disc,Img,bloodVessel,xc,yc] = discExtractSmoothMuch(inImg)
    U=inImg(:,:,1);
    G=inImg(:,:,2);
    mask=better_bloodVesselExtract(G);
    bloodVessel = mask;
    U(find(U<30))=90;
    hsvImg=rgb2hsv(inImg);
    V=hsvImg(:,:,3);
    V=grayStretch(V);
    V=FastInpaint(V,mask,500);
    U=FastInpaint(U,mask,500);
    G=FastInpaint(G,mask,500);
    % get the size
    [nrow,ncol] =size(U);
    ic=nrow/2;
    jc=ncol/2;
    r=round(nrow/3);
    phi_0 = sdf2circle(nrow,ncol,ic,jc,r);
    delta_t = 5;
    lambda_1=1;
    lambda_2=1;
    nu=0;
    h = 1;
    epsilon=8;
    mu = 0.01*255*255;
    I=U;
    % iteration should begin from here
    phi=phi_0;
    numIter = 10;
    for k=1:70
        phi=evolution_cv(I, phi, mu, nu, lambda_1, lambda_2, delta_t, epsilon, numIter); % update level set function
    end
    %%
    phi=im2bw(phi,0);
    phi=bwareaopen(phi,100);
    phi=imcomplement(phi);
    phi=bwareaopen(phi,200);
    edge_disc=edge(phi);
    area=find(phi==1);
    se= strel('disk',1);
    edge_disc = imdilate(edge_disc,se);
    %%
    [m,n]=size(edge_disc);
    cc = bwconncomp(edge_disc);
    numFields = getfield(cc,'NumObjects');
    if(numFields > 1)
        S = regionprops(cc, 'Area');
        P = max([S.Area]);
        L = labelmatrix(cc);
        temp = ismember(L, find([S.Area] >= P));
        for i = 1:m
            for j = 1:n
                if(temp(i,j) == 0)
                    edge_disc(i,j) = 0;
                end
            end
        end
    end
    %%
    dif=4;
    se=strel('disk',1);
    db=0;
    while(dif>3)
        [edge_disc,dif]=eliminateBlank(edge_disc);
        edge_disc=imclose(edge_disc,se);
        db=db+1;
        if(db==10)
            break;
        end
    end
    edge_disc=imdilate(edge_disc,se);
    edge_disc=edgeOptimize(edge_disc,10);
    phi=eTp(edge_disc);
    % phi is mask
    %figure, imshow(phi)
    edge_disc=edge(phi);
    % edge of the final disc image
    %figure, imshow(edge_disc)
    [x,y]=find(edge_disc>0);
    xc=round((min(x)+max(x))/2);
    yc=round((min(y)+max(y))/2);
    num=size(find(phi==1),1);
    Img=inImg;
    [x]=find(edge_disc>0);
    r=Img(:,:,1);
    g=Img(:,:,2);
    b=Img(:,:,3);
    r(x)=0;
    g(x)=255;
    b(x)=0;
    Img(:,:,1)=r;
    Img(:,:,2)=g;
    Img(:,:,3)=b;
    Img(xc-1:xc+1,yc-1:yc+1,1)=0;
    Img(xc-1:xc+1,yc-1:yc+1,2)=255;
    Img(xc-1:xc+1,yc-1:yc+1,3)=0;
    %figure, imshow(Img)
end

function [edge_disc]=edgeOptimize(edge_disc,iteration)
    [m,n]=size(edge_disc);
    [x,y]=find(edge_disc>0);
    xc=round((min(x)+max(x))/2);
    yc=round((min(y)+max(y))/2);
    rad=ones(size(x,1),1);
    ang=ones(size(rad));
    for l=1:size(x,1)
        delta_x=x(l)-xc;
        delta_y=y(l)-yc;
        rad(l)=sqrt(delta_x*delta_x+delta_y*delta_y);
        cos1=delta_x/rad(l);
        ang(l)=acosd(cos1);
        if(delta_y<0)
            ang(l)=360-ang(l);
        end
    end
    %%
    [sortAng,ind]=sort(ang);
    diffang=diff(sortAng);
    sortRad=ones(size(sortAng));
    
    for l=1:size(ind,1)
        sortRad(l)=rad(ind(l));
    end
    
    sortRad1=sortRad;
    
    for j=1:iteration
        [sortRad1]=eliminatePeak1(sortRad1,sortAng);
    end
    
    delta_rad=sortRad1-sortRad;
    edge_disc(find(edge_disc==1))=0;
    delta_x=floor(delta_rad.*cosd(sortAng));
    delta_y=floor(delta_rad.*sind(sortAng));
    
    for l=1:size(ind,1)
        x(ind(l))=x(ind(l))+delta_x(l);
    end
    for l=1:size(ind,1)
        y(ind(l))=y(ind(l))+delta_y(l);
    end
    
    x(find(x<=1))=1;
    y(find(y<=1))=1;
    x(find(x>=m))=m-1;
    y(find(y>=n))=n-1;
    
    for l=1:size(x,1)
        edge_disc(x(l),y(l))=1;
    end    
end

function [addedge,dif]=eliminateBlank(edge_disc)
    [x,y]=find(edge_disc>0);
    [m,n]=size(edge_disc);
    xc=round((min(x)+max(x))/2);
    yc=round((min(y)+max(y))/2);
    rad=ones(size(x,1),1);
    ang=ones(size(rad));
    for l=1:size(x,1)
        delta_x=x(l)-xc;
        delta_y=y(l)-yc;
        rad(l)=sqrt(delta_x*delta_x+delta_y*delta_y);
        cos1=delta_x/rad(l);
        ang(l)=acosd(cos1);
        if(delta_y<0)
            ang(l)=360-ang(l);
        end
    end
    %%
    
    [sortAng,ind]=sort(ang);
    sortRad=ones(size(sortAng));
    for l=1:size(ind,1)
        sortRad(l)=rad(ind(l));
    end
    diffAng=abs(diff(sortAng));
    dif1=sortAng(1)+360-sortAng(end);
    mean_diffAng=mean(diffAng);
    [dif,maxInd]=max(diffAng);
    if(dif1>dif)
        dif=dif1;
        addNum=floor(dif/mean_diffAng);
        meanDiffRad=(sortRad(1)-sortRad(end))/addNum;
        addAng=zeros(size(ind,1)+addNum,1);
        addRad=zeros(size(addAng));
        addAng(1:size(ind,1),1)=ang;
        addRad(1:size(ind,1),1)=rad;
        or=size(ind,1);
        for count=1:addNum
            tem=(sortAng(end)+mean_diffAng*count);
            if(tem>360)
                tem=tem-360;
            end
            addAng(or+count,1)=tem;
            addRad(or+count,1)=sortRad(end)+meanDiffRad*count;
        end
        addx=zeros(size(ind,1)+addNum,1);
        addy=zeros(size(ind,1)+addNum,1);
        addx(1:size(ind,1),1)=x;
        addy(1:size(ind,1),1)=y;
        for l=or+1:size(addAng,1)
            delta_x=round(addRad(l)*cosd(addAng(l)));
            delta_y=round(addRad(l)*sind(addAng(l)));
            addx(l)=xc+delta_x;
            addy(l)=yc+delta_y;
        end
        addx(find(addx<=1))=1;
        addy(find(addy<=1))=1;
        addx(find(addx>=m))=m-1;
        addy(find(addy>=n))=n-1;
        addedge=zeros(size(edge_disc));
        for k=1:size(addx,1)
            addedge(addx(k),addy(k))=1;
        end
        
    %%
    elseif(~isnan(mean_diffAng))
        addNum=floor((sortAng(maxInd+1)-sortAng(maxInd))/mean_diffAng);
        meanDiffRad=(sortRad(maxInd+1)-sortRad(maxInd))/addNum;
        addAng=zeros(size(ind,1)+addNum,1);
        addRad=zeros(size(addAng));
        addAng(1:size(ind,1),1)=ang;
        addRad(1:size(ind,1),1)=rad;
        or=size(ind,1);
        for count=1:addNum
            addAng(or+count,1)=sortAng(maxInd)+mean_diffAng*count;
            addRad(or+count,1)=sortRad(maxInd)+meanDiffRad*count;
        end
        addx=zeros(size(ind,1)+addNum,1);
        addy=zeros(size(ind,1)+addNum,1);
        addx(1:size(ind,1),1)=x;
        addy(1:size(ind,1),1)=y;
        for l=or+1:size(addAng,1)
            delta_x=round(addRad(l)*cosd(addAng(l)));
            delta_y=round(addRad(l)*sind(addAng(l)));
            addx(l)=xc+delta_x;
            addy(l)=yc+delta_y;
        end
        addx(find(addx<=1))=1;
        addy(find(addy<=1))=1;
        addx(find(addx>=m))=m-1;
        addy(find(addy>=n))=n-1;
        addedge=zeros(size(edge_disc));
        for k=1:size(addx,1)
            addedge(addx(k),addy(k))=1;
        end
    
    else
        addedge=edge_disc;
    end
    
end

function [y]=eliminatePeak1(y,x)
    numP=zeros(180,1);
    difmean=zeros(180,1);
    sumDif=zeros(180,1);
    count=0;
    for ang=2:2:360
        count=count+1;
        [numP(count,1),difmean(count,1),sumDif(count,1)]=numBy(ang-2,ang,x,y);
    end
    % figure,plot(1:180,numP);
    [maxNum,inMax]=max(difmean);
    
    for i=15:-1:1
        ran=-i:1:i;
        ran=ran+inMax;
        tem=find(ran<1);
        ran(tem)=ran(tem)+count;
        tem=find(ran>180);
        ran(tem)=ran(tem)-count;
        numRan=2*i+1;
        meanNum=sum(numP(ran))/numRan;
        meandif=sum(difmean(ran))/numRan;
        if(meanNum>4 & meandif>0.7)
            ind1=ran(1)*2;
            [~,ind1]=min(abs(x-ind1));
            ind2=ran(2*i+1)*2;
            [~,ind2]=min(abs(x-ind2));
            rad1=y(ind1);
            rad2=y(ind2);
            increase=(rad2-rad1)/(numRan-1);
            for j=2:size(ran,2)
                if ran(j)==1
                    changeInd=find(x>=(ran(j)*2-2)&x<=(ran(j)*2));
                else
                    changeInd=find(x>(ran(j)*2-2)&x<=(ran(j)*2));
                end
                y(changeInd)=increase*(j-1)+rad1;
            end
            break;
        end    
    end    
end
function [numP,difMean,sumDif]=numBy(ang1,ang2,x,y)
    n=size(x,1);
    dif=diff(y);
    dif=abs(dif);
    if ang1==0
        ind=find(x>=ang1&x<=ang2);
    else
        ind=find(x>ang1&x<=ang2);
    end
    numP=size(ind,1);
    sumDif=0;
    for k=1:size(ind,1)
        in=ind(k);
        if in==k
            sumDif=sumDif+abs(y(end)-y(1));
        else
            sumDif=sumDif+dif(in-1);
        end
    end
    difMean=sumDif/numP;    
end

function [phi]=eTp(edge_disc)
    phi=zeros(size(edge_disc));
    [x,y]=find(edge_disc>0);
    for i=min(x):1:max(x)
        indx=find(x==i);
        y1=min(y(indx));
        y2=max(y(indx));
        for j=y1:1:y2
            phi(i,j)=1;
        end
    end
    se=strel('disk',2);
    phi=imclose(phi,se);
end

function u = EVOLUTION(u0, g, lambda, mu, alf, epsilon, delt, numIter)
    % EVOLUTION(u0, g, lambda, mu, alf, epsilon, delt, numIter) updates the level set function
    % according to the level set evolution equation in Chunming Li et al's paper:
    % "Level Set Evolution Without Reinitialization: A New Variational Formulation"
    % in Proceedings CVPR'2005,
    % Usage:
    % u0: level set function to be updated
    % g: edge indicator function
    % lambda: coefficient of the weighted length term L(\phi)
    % mu: coefficient of the internal (penalizing) energy term P(\phi)
    % alf: coefficient of the weighted area term A(\phi), choose smaller alf
    % epsilon: the papramater in the definition of smooth Dirac function, default value 1.5
    % delt: time step of iteration, see the paper for the selection of time step and mu
    % numIter: number of iterations.
    %
    % Author: Chunming Li, all rights reserved.
    % e-mail: li_chunming@hotmail.com
    % http://vuiis.vanderbilt.edu/~licm/
    u=u0;
    [vx,vy]=gradient(g);
    for k=1:numIter
        u=NeumannBoundCond(u);
        [ux,uy]=gradient(u);
        normDu=sqrt(ux.^2 + uy.^2 + 1e-10);
        Nx=ux./normDu;
        Ny=uy./normDu;
        diracU=Dirac(u,epsilon);
        K=curvature_central(Nx,Ny);
        weightedLengthTerm=lambda*diracU.*(vx.*Nx + vy.*Ny + g.*K);
        penalizingTerm=mu*(4*del2(u)-K);
        weightedAreaTerm=alf.*diracU.*g;
        u=u+delt*(weightedLengthTerm + weightedAreaTerm + penalizingTerm); % update the level set function
    end
end
% the following functions are called by the main function EVOLUTION
function f = Dirac(x, sigma)
    f=(1/2/sigma)*(1+cos(pi*x/sigma));
    b = (x<=sigma) & (x>=-sigma);
    f = f.*b;
end
function K = curvature_central(nx,ny)
    [nxx,junk]=gradient(nx);
    [junk,nyy]=gradient(ny);
    K=nxx+nyy;
end
function g = NeumannBoundCond(f)
    % Make a function satisfy Neumann boundary condition
    [nrow,ncol] = size(f);
    g = f;
    g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);
    g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);
    g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);  
end

function phi = EVOLUTION_CV(I, phi0, mu, nu, lambda_1, lambda_2, delta_t, epsilon, numIter);
    % evolution_withoutedge(I, phi0, mu, nu, lambda_1, lambda_2, delta_t, delta_h, epsilon, numIter);
    % input:
    % I: input image
    % phi0: level set function to be updated
    % mu: weight for length term
    % nu: weight for area term, default value 0
    % lambda_1: weight for c1 fitting term
    % lambda_2: weight for c2 fitting term
    % delta_t: time step
    % epsilon: parameter for computing smooth Heaviside and dirac function
    % numIter: number of iterations
    % output:
    % phi: updated level set function
    %
    % created on 04/26/2004
    % author: Chunming Li
    % email: li_chunming@hotmail.com
    % Copyright (c) 2004-2006 by Chunming Li
    I=BoundMirrorExpand(I);
    phi=BoundMirrorExpand(phi0);
    for k=1:numIter
        phi=BoundMirrorEnsure(phi);
        delta_h=Delta(phi,epsilon);
        Curv = curvature(phi);
        [C1,C2]=binaryfit(phi,I,epsilon);
        % updating the phi function
        phi=phi+delta_t*delta_h.*(mu*Curv-nu-lambda_1*(I-C1).^2+lambda_2*(I-C2).^2);
    end
    phi=BoundMirrorShrink(phi);
end

function [filledDisc,edge_disc,edge_disc1,bloodVessel,xc,yc] = extractDisc(inImg,pixl)
    % example that CV model works well
    G=inImg(:,:,2);
    G(find(G<30))=60;
    mask=better_bloodVesselExtract(G);
    bloodVessel = mask;
    U=inImg(:,:,1);
    U(find(U<30))=90;
    U=FastInpaint(U,mask,500);
    se=strel('disk',10);
    % U=imopen(U,se);
    %I=['inpaintV',num2str(imgNum),'.tif'];
    th=graythresh(U);
    bw_th=im2bw(U,th);
    edge_bw=edge(bw_th);
    U=grayStretch(U);
    %imwrite(U,I);
    % get the size
    [nrow,ncol] =size(U);
    ic=nrow/2;
    jc=ncol/2;
    r=100;
    phi_0 = sdf2circle(nrow,ncol,ic,jc,r);
    delta_t = 0.1;
    lambda_1=1;
    lambda_2=1;
    nu=0;
    h = 1;
    epsilon=8;
    mu = 0.01*255*255;
    I=U;
    % iteration should begin from here
    phi=phi_0;
    numIter = 10;
    for k=1:70
        phi=evolution_cv(I, phi, mu, nu, lambda_1, lambda_2, delta_t, epsilon, numIter); % update level set function
    end
    %%
    phi=im2bw(phi,0);
    phi=bwareaopen(phi,100);
    phi=imcomplement(phi);
    phi=bwareaopen(phi,200);
    edge_disc=edge(phi);
    area=find(phi==1);
    se= strel('disk',1);
    edge_disc = imdilate(edge_disc,se);
    %%
    [m,n]=size(edge_disc);
    cc = bwconncomp(edge_disc);
    numFields = getfield(cc,'NumObjects');
    if(numFields > 1)
        S = regionprops(cc, 'Area');
        P = max([S.Area]);
        L = labelmatrix(cc);
        temp = ismember(L, find([S.Area] >= P));
        for i = 1:m
            for j = 1:n
                if(temp(i,j) == 0)
                    edge_disc(i,j) = 0;
                end
            end
        end
    end
    %%
    [edge_disc,edge_disc1]=edgeOptimize(edge_disc,pixl);
    [edge_disc1,~]=edgeOptimize(edge_disc1,0);
    [x,y]=find(edge_disc>0);
    xc=round(mean(x));
    yc=round(mean(y));
    filledDisc = FillDisc(edge_disc1);
end

function k = FastInpaint(I,mask,liter)%(I,mask,liter,bar)
    %Image inpainting by Manuel M. Oliveira's method Fast Digital Image
    %Inpainting
    % 09-10-2007
    %parameters:
    % I------image to be inpainted
    % mask------the noise mask which is a binary image
    % liter------literation times
    % bar______diffusion barrier
    %timer start
    I=im2double(I);
    if islogical(mask)
        mask=mask;
    else
        mask=im2bw(mask);
    end
    diffker=[0.073235 0.176765 0.0732325; 0.17675 0 0.17675; 0.073235 0.176775 0.073235];
    %There are two kinds of diffusion kernels in Olivera's article I use the
    %first one[0.073235 0.176765 0.0732325; 0.17675 0 0.17675; 0.073235 0.176775 0.073235].
    %Another is [0.125 0.125 0.125; 0.125 0 0.125; 0.125 0.125 0.125].
    [r,c]=find(mask);
    if (size(I, 3) == 3) %Color image process
        r=FInpaintGray(I(:,:,1),mask,liter,diffker);
        g=FInpaintGray(I(:,:,2),mask,liter,diffker);
        b=FInpaintGray(I(:,:,3),mask,liter,diffker);
        k=cat(3,r,g,b);
    else %Gray image process
        k=FInpaintGray(I,mask,liter,diffker);
    end
    k=uint8(k.*255);
    %show the elapsed time.
end
function g = FInpaintGray(I,mask,liter,diffker)
    [r,c]=find(mask);
    f=incrSize(I);
    for n=1:liter
        for i=1:length(r)
            x=r(i)+1;%+1
            y=c(i)+1;%+1
            f(x,y)=f(x-1,y-1)*0.073235 + f(x-1,y)*0.176765 + f(x-1,y+1)*0.073235 + f(x,y-1)*0.176765 +f(x,y+1)*0.176765 + f(x+1,y-1)*0.073235 + f(x+1,y)*0.176765 + f(x+1,y+1)*0.073235;
            %f(x,y)=f(x-1,y-1)*diffker(1,1)+f(x-1,y)*diffker(1,2)+f(x-1,y+1)*diffker(1,3)+f(x,y-1)*diffker(2,1)+f(x,y+1)*diffker(2,3)+f(x+1,y-1)*diffker(3,1)+f(x+1,y)*diffker(3,2)+f(x+1,y+1)*diffker(3,3);
        end
    end
    g=mat2gray(f(2:end-1,2:end-1));
end
function u = incrSize(f)
    A=im2double(zeros(size(f,1)+2,size(f,2)+2));
    A(2:end-1,2:end-1)=f;
    A(1:1,2:end-1)=f(1:1,1:end);
    A(2:end-1,1:1)=f(1:end,1:1);
    A(2:end-1,end:end)=f(1:end,end:end);
    A(end:end,2:end-1)=f(end:end,1:end);
    u=A;
end

function [Img_ellipse,h,k,a,b,x_set] = FillEllipse(cx,cy,r,C,M,N,Eye_side,e)
    %C = correction on up, down, right and left
    %e = correction on the side where there is the blood vessels
    Img_ellipse = zeros(M,N);
    p_up = max(0,round(cy-r-C(1)*r));
    p_down = min(M,round(cy+r+C(2)*r));
    p_left = max(0,round(cx-r-C(4)*r));
    p_right = min(N,round(cx+r+C(3)*r));
    if Eye_side=='R'
        p_right = min(N,p_right+round(r*e));
    else
        p_left = max(0,round(p_left-e*r));
    end
    a = round(abs(p_right - p_left)/2);
    b = round(abs(p_down - p_up)/2);
    h = round(p_left + (p_right - p_left)/2);
    k = round(p_up + (p_down - p_up)/2);
    x_set = h-a:1:h+a;
    [x,y]=meshgrid(-(h-1):(N-h),-(k-1):(M-k));
    Img_ellipse =((x.^2/a^2+y.^2/b^2)<=1);
    %figure(),imshow(Img_ellipse);
end

function [Edges,I,opt_square,max_pixl] = FindBloodVesselPoint2(M,N,BloodVessel,L,cx,cy,Eye_side)
    Edges = zeros(M,N);
    [x,y]=meshgrid(1:N,1:M);
    uperHalfL = round(L/2);
    downHalfL = L - uperHalfL;
    flag = 0;
    x1 = cx;
    y1 = max(0,cy-uperHalfL);
    y2 = y1+L;
    if Eye_side == 'R'
        a = 1;
    else
        a = -1;
    end
    max_pixl = zeros(1,0);
    while(flag==0)
        Img_square = zeros(M,N);
        x2 = x1+a*L;
        if x2>N || x2<0
            flag = 1;
        else
            xx1 = min(x1,x2);
            xx2 = max(x1,x2);
            for i=1:1:M
                for j=1:1:N
                    if x(i,j)>=xx1 && x(i,j)<=xx2 && y(i,j)>=y1 && y(i,j)<=y2
                        %Img_square(j,i) = 1;
                        Img_square(i,j) = 1;
                    end
                end
            end
            Edges = Edges + edge(Img_square,'sobel');
            point = BloodVessel.*Img_square;
            max_pixl = [max_pixl,nnz(point)];
            x1 = x2;
        end
    end
    [~,I] = max(max_pixl);
    opt_square = zeros(M,N);
    x1 = cx+a*L*(I-1);
    x2 = x1+a*L;
    xx1 = min(x1,x2);
    xx2 = max(x1,x2);
    for i=1:1:M
        for j=1:1:N
            if x(i,j)>=xx1 && x(i,j)<=xx2 && y(i,j)>=y1 && y(i,j)<=y2
                opt_square(i,j) = 1;
            end
        end
    end
end

function [fdy,fdx]=forward_gradient(f)
    % function [fdx,fdy]=forward_gradient(f);
    %
    % created on 04/26/2004
    % author: Chunming Li
    % email: li_chunming@hotmail.com
    % Copyright (c) 2004-2006 by Chunming Li
    [nr,nc]=size(f);
    fdx=zeros(nr,nc);
    fdy=zeros(nr,nc);
    a=f(2:nr,:)-f(1:nr-1,:);
    fdx(1:nr-1,:)=a;
    b=f(:,2:nc)-f(:,1:nc-1);
    fdy(:,1:nc-1)=b;
end

%this function computes the phi required initially */
function [xcontour, ycontour] = get_phi(I, nrow, ncol,margin)
    % I is the image matrix
    % nrow is the no of rows
    % ncol is the no of columns
    count=1;
    x=margin;
    for y=margin:nrow-margin+1
        xcontour(count) = x;
        ycontour(count) = y;
        count=count+1;
    end
    y=nrow-margin+1;
    for x=margin+1:ncol-margin+1
        xcontour(count) = x;
        ycontour(count) = y;
        count=count+1;
    end
    x=ncol-margin+1;
    for y=nrow-margin:-1:margin
        xcontour(count) = x;
        ycontour(count) = y;
        count=count+1;
    end
    y=margin;
    for x=ncol-margin:-1:margin+1
        xcontour(count) = x;
        ycontour(count) = y;
        count=count+1;
    end
end

function[B]= grayStretch(Ac)
    M=max(max(Ac));
    m=min(min(Ac));
    B=double(Ac-m);
    B=B/double(M-m)*double(255-0);
    B=uint8(B);
end

function H = Heaviside(phi,epsilon)
    % Heaviside(phi,epsilon) compute the smooth Heaviside function
    %
    % created on 04/26/2004
    % author: Chunming Li
    % email: li_chunming@hotmail.com
    % Copyright (c) 2004-2006 by Chunming Li
    H = 0.5*(1+ (2/pi)*atan(phi./epsilon));
end

function Xcol_seg = localization(I)
    trim_pxl = 100;
    inputpixel=175;
    I_red=I(:,:,1);
    if length(size(I))==3
        I_bw = rgb2gray(I);
    end
    [M, N, O] = size(I);
    
    % global p;
    % global L;
    % global Th;
    % global alpha;
    %
    % global max_val;
    % global min_val;
    L = 256;
    H=imhist(I_bw);
    
    %figure,imhist(I)
    p = H / (M * N);
    
    %figure(2); plot(h)
    max_val=double(max(max(I_red)));
    min_val=double(min(min(I_red))+1);
    
    %*********************************************************************
    % 1. Define problem hyperspace and plot in 2D
    %*********************************************************************
    Th =2;
    % No of thresholds
    D = Th*2; % no of dimensions
    range_min = min_val*ones(1,D);
    range_max = max_val*ones(1,D); % minimum & maximum range;
    disp(range_max)
    alpha=1.5;
    %*********************************************************************
    % 2. initialize the population
    %*********************************************************************
    NP = 10*D ; % population size
    maxgen =100; % no of generations
    F = 0.5;
    CR = 0.9;
    max_runs = 2;
    globalbest1 = [];
    statistics_f = [];
    statistics_x = [];
    % tstart = tic;
    for runn = 1:max_runs
        x=[];
        for i = 1 : NP
            for j = 1:D
                x(i, j) = round(range_min(j) + ((range_max(j)-range_min(j))*(rand)));
            end
            x(i,:)=sort(x(i,:));
            fitness_parent(i,1) = ultrafuzziness([0 0 x(i,:) 255 255],p,alpha);
        end
        v = zeros(size(x));
        u = zeros(size(x));
        %*********************************************************************
        % 4. start iteration
        %*********************************************************************
        % tStart = tic;
        for gen = 2:maxgen
            % [o-2]
            %*********************************************************************
            % 3. find mutation population
            %*********************************************************************
            for i = 1:NP
                r = ceil(rand(1,3)*NP);
                while r(1)==r(2) || r(2)== r(3) || min(r)==0 || max(r)>NP
                    r = ceil(rand(1,3)*NP);
                end
                
                v(i,:) = x(r(1),:) + F*(x(r(2),:) - x(r(3),:));
                for j = 1:D
                    if rand > CR
                        u(i,j) = x(i,j);
                    else
                        u(i,j) = v(i,j);
                    end
                end
                u(i,:)= round(u(i,:));
                u(i,:)=sort(u(i,:));
            end
            for i = 1:NP
                for jj = 1:D
                    u(i,jj) = max(u(i,jj), range_min(jj));
                    u(i,jj) = min(u(i,jj), range_max(jj));
                end
                u(i,:)=sort(u(i,:));
                fitness_child(i,1) = ultrafuzziness([0 0 u(i,:) 255 255],p,alpha);
            end
            for i = 1:NP
                if fitness_parent(i) < fitness_child(i)
                    fitness_parent(i) = fitness_child(i);
                    x(i,:) = u(i,:);
                end
            end
            [globalbest,globalbest_index] = max(fitness_parent);
            global_xbest = sort(x(globalbest_index,:));

            % clc
            % runn
            % fprintf('Optimisation through Differential Evolution\n')
            % fprintf('Generation: %0.5g\nGlobalbest: %2.7g\n', gen, globalbest)
            % fprintf('Best particle position : %0.11g\n', global_xbest)
            globalbest1 = [globalbest1, globalbest];
        end
        % tElapsed = toc(tStart);
        % tElapsed
        globalbest1 = [globalbest1, globalbest];
        statistics_f = [statistics_f, globalbest];
        statistics_x = [statistics_x; (global_xbest)];
    end
    %plot(1:NP:NP*50,globalbest1(1:50),'-bs','MarkerFaceColor','b');
    %hold on
    f_mean = mean(statistics_f);
    f_stddev = std(statistics_f);
    best_fitness = max(statistics_f);
    worst_fitness = min(statistics_f);
    x_median = median(statistics_x);
    %% select the threshold points
    % T = [min_val round(x_median) max_val];
    % Thres=T(2:2:2*Th)
    t1=x_median(1:2:D);
    t2=x_median(2:2:D);
    Thres=round((t1+t2)/2);
    X=grayslice(I_bw,[Thres]);
    X1=uint8(255*mat2gray(X));
    % timestop=toc(tstart);
    % tstop1(im_num)=timestop/2;
    %figure,imshow(X1);
    %% Trimp the image first
    XX1 = Trimp2(X1,trim_pxl);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %XX1 = X1;
    %% New Code cup is the largest spot
    X3 = XX1==255;
    % Find all the connected components
    CC = bwconncomp(X3);
    % Number of pixels in each connected components
    numPixels = cellfun(@numel,CC.PixelIdxList);

    % Largest connected compnent
    [biggest,idx] = max(numPixels);
    % if(aaaa)
    % %% Additional code to remove Fringe
    % numPixels_temp = sort(numPixels,'descend');
    % idx = find(numPixels==numPixels_temp(2));
    % end
    %Calculating the cetroid of the largest component

    S = regionprops(CC,'Centroid'); % Calculate centroids for connected components in the image using regionprops.
    cntr = cat(1, S.Centroid); %Concatenate structure array containing centroids into a single matrix.
    centroid_x=round(cntr(idx,1));
    centroid_y=round(cntr(idx,2));
    
    %figure, imshow(I)
    % hold on
    % plot(centroid_x,centroid_y, 'b*')
    % hold off
    % Calculating region

    newx_up=centroid_y-inputpixel;
    newx_down=min(M,centroid_y+inputpixel);
    newy_left=centroid_x-inputpixel;
    newy_right=min(N,centroid_x+inputpixel);
    % tstop2(im_num)=toc(tstart);

    %% Extract image
    for i = newx_up :newx_down
        for j = newy_left : newy_right
            Xcol_seg(i - newx_up + 1, j - newy_left + 1,:) = I(i,j,:);
        end
    end
    % clear p;
    % clear L; 
    % clear Th;
    % clear alpha;
    %
    % clear max_val;
    % clear min_val;
end

function [Img_ellipse] = MakeFilledEllipse2(M,N,h,k,a,b,ch,ck)
    Img_ellipse = zeros(M,N);
    h = h+ch;
    k = k+ck;
    [x,y]=meshgrid(-(h-1):(N-h),-(k-1):(M-k));
    Img_ellipse =((x.^2/a^2+y.^2/b^2)<=1);
end

function T = maxFilter(inImg , w)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % Computes the maximum 'local' dynamic range
    %
    % inImg : grayscale image
    % [-w, w]^2 : search window (w must be odd)
    % T : maximum local dynamic range
    %
    % Author: Kunal N. Chaudhury
    % Date: March 1, 2012
    %
    % Reference:
    %
    % K.N. Chaudhury, "Acceleration of the shiftable O(1) algorithm for
    % bilateral filtering and non-local means," arXiv:1203.5128v1.
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    T = -1;
    sym = (w - 1)/2;
    [m, n] = size(inImg);
    pad1 = w*ceil(m/w) - m;
    pad2 = w*ceil(n/w) - n;
    inImg = padarray(inImg, [pad1 pad2], 'symmetric', 'post');
    template = inImg;
    m = m + pad1;
    n = n + pad2;
    % scan along row
    for ii = 1 : m
        L = zeros(n, 1);
        R = zeros(n, 1);
        L(1) = template(ii, 1);
        R(n) = template(ii, n);
        for k = 2 : n
            if mod(k - 1, w) == 0
                L(k) = template(ii , k);
                R(n - k + 1) = template(ii , n - k + 1);
            else
                L(k) = max( L(k-1) , template(ii, k) );
                R(n - k + 1) = max( R(n - k + 2), template(ii, n - k + 1) );
            end
        end
        for k = 1 : n
            p = k - sym;
            q = k + sym;
            if p < 1
                r = -1;
            else
                r = R(p);
            end
            if q > n
                l = -1;
            else
                l = L(q);
            end
            template(ii, k) = max(r,l);
        end
    end
    % scan along column
    for jj = 1 : n
        L = zeros(m, 1);
        R = zeros(m, 1);
        L(1) = template(1, jj);
        R(m) = template(m, jj);
        for k = 2 : m
            if mod(k - 1, w) == 0
                L(k) = template(k, jj);
                R(m - k + 1) = template(m - k + 1, jj);
            else
                L(k) = max( L(k - 1), template(k, jj) );
                R(m - k + 1) = max( R(m - k + 2), template(m - k + 1, jj));
            end
        end
        for k = 1 : m
            p = k - sym;
            q = k + sym;
            if p < 1
                r = -1;
            else
                r = R(p);
            end
            if q > m
                l = -1;
            else
                l = L(q);
            end
            temp = max(r,l) - inImg(k, jj);
            if temp > T
                T = temp;
            end
        end
    end
end

function [xMin,yMin,xMax,yMax,vDistance,hDistance] = maxminPoints(inImg)
    %figure, imshow(inImg)
    [y,x] = find(inImg>=1);
    xMax = max(x);
    xMin = min(x);
    yMax = max(y);
    yMin = min(y);
    vDistance = yMax - yMin;
    hDistance = xMax - xMin;
    inImg(yMax,:)=1;
    inImg(yMin,:)=1;
    inImg(:,xMax) = 1;
    inImg(:,xMin) = 1;
    %figure,imshow(inImg)
end

function [centers, radii] = PaintCircle(in,e)
    [M,N] = size(in);
    % edges = edge(in,'sobel');
    % spot_size = 25;
    % edges = bwareaopen(edges,spot_size);%figure;imshow(x)
    spot_size = 50;
    in = bwareaopen(in,spot_size);%figure;imshow(x)
    se = strel('disk',10);
    in = imclose(in,se);
    edges = edge(in,'sobel');
    in = [];
    in = edges;
    M1 = 0;
    M2 = 0;
    for i=1:1:M
        for j = 1:1:N
            if in(i,j)>0
                M1 = i;
                j = N;
                i = M;
            end
        end
    end
    for i=M:-1:1
        for j = 1:1:N
            if in(i,j)>0
                M2 = i;
                j = N;
                i = M;
            end
        end
    end
    if M1*M2>0
        r = round(abs(M1-M2)/2);
    else
        r = 60;
    end
    edges = edge(in,'sobel');
    [centers, radii, ~] = imfindcircles(edges,[r-e,r+e],'Sensitivity',0.9775,'Method','twostage');
    %hold on
    %figure, imshow(edges);
    %hold off
    %h = viscircles(centers,radii);
    % hold on
    %h = viscircles([centers(1,1),centers(1,2)],radii(1));
    % hold off
    % hold on
    % h = viscircles([centers(2,1),centers(2,2)],radii(2));
    % hold off
end
function [gray]=PCA_rgb2gray(rgb)
    [row,col,dim]=size(rgb);
    A=zeros(dim,row*col);
    count=1;
    for i=1:row
        for j=1:col
            A(:,count)=rgb(i,j,:);
            count=count+1;
        end
    end
    [x,~,la]=pca(A');
    eV=[x(:,3),x(:,2),x(:,1)];
    %PC1
    P_e=eV(:,3);
    AI=P_e'*A;
    count=1;
    for i=1:row
        for j=1:col
            I(i,j)=AI(:,count);
            count=count+1;
        end
    end
    Ac=I;
    M=max(max(Ac));
    m=min(min(Ac));
    B=double(Ac-m);
    B=B/double(M-m)*double(255-0);
    B=uint8(B);
    gray=B;
end

function [c,h]=plotLevelSet(u,zLevel, style)
    % plotLevelSet(u,zLevel, style) plot the level contour of function u at
    % the zLevel.
    % created on 04/26/2004
    % author: Chunming Li
    % email: li_chunming@hotmail.com
    % Copyright (c) 2004-2006 by Chunming Li
    % hold on;
    [c,h] = contour(u,[zLevel zLevel],style);
    % hold off;
end

function f = sdf2circle(nrow,ncol, ic,jc,r)
    % sdf2circle(nrow,ncol, ic,jc,r) computes the signed distance to a circle
    % input:
    % nrow: number of rows
    % ncol: number of columns
    % (ic,jc): center of the circle
    % r: radius of the circle
    % output:
    % f: signed distance to the circle
    %
    % created on 04/26/2004
    % author: Chunming Li
    % email: li_chunming@hotmail.com
    % Copyright (c) 2004-2006 by Chunming Li
    [X,Y] = meshgrid(1:ncol, 1:nrow);
    f = sqrt((X-jc).^2+(Y-ic).^2)-r;
    % figure;
    % imagesc(f)
end

function u = signed_distance(I,xcontour, ycontour,margin)
    % I is the image matrix
    % nrow is the no of rows
    % ncol is the no of columns
    [nrow, ncol] = size(I);
    [temp, contsize] = size(xcontour);
    Mark = zeros(nrow, ncol);
    for y=1:nrow
        for x=1:ncol
            if (x > ncol-margin+1) | (x < margin) | (y < margin) | (y > nrow-margin+1)
                Mark(y,x) = -1;
            end
        end
    end
    for y = 1:nrow
        for x =1: ncol
            u(y,x) = sqrt(min((x-xcontour).^2+(y-ycontour).^2));
            if Mark(y,x) == -1    
                u(y,x) = -u(y,x);
            end
        end
    end
end

function [Thres,X1] = Thresholding(I1,Th)
    %figure, imshow(I1)
    I=I1(:,:,1:3);
    I_red=I(:,:,1);
    %%%%%%%
    A=I_red;
    M=max(max(A));
    m=min(min(A));
    B=double(A-m);
    B=B/double(M-m)*double(255-0);
    B=uint8(B);
    I_red=B;
    %%%%%%%%%%%
    if length(size(I))==3
        I_bw = rgb2gray(I);
    end
    [M, N, O] =size(I);
    L = 256;
    %H=imhist(I_bw);
    H=imhist(I(:,:,2));
    %figure,imhist(I)
    p = H / (M * N);
    %figure(2); plot(h)
    max_val=double(max(max(I_red)));
    min_val=double(min(min(I_red))+1);
    %*********************************************************************
    % 1. Define problem hyperspace and plot in 2D
    %*********************************************************************
    %Th =7;
    % No of thresholds
    D = Th*2; % no of dimensions
    range_min = min_val*ones(1,D);
    range_max = max_val*ones(1,D);% minimum & maximum range;
    alpha=1.5;
    %*********************************************************************
    % 2. initialize the population
    %*********************************************************************
    NP = 10*D ; % population size
    maxgen =100; % no of generations
    F = 0.5;
    CR = 0.9;
    max_runs = 2;
    globalbest1 = [];
    statistics_f = [];
    statistics_x = [];
    tstart = tic;
    for runn = 1:max_runs
        x=[];
        for i = 1 : NP
            for j = 1:D
                x(i, j) = round(range_min(j) + ((range_max(j)-range_min(j))*(rand)));
            end
            x(i,:)=sort(x(i,:));
            fitness_parent(i,1) = ultrafuzziness([0 0 x(i,:) 255 255],p,alpha);
        end
        v = zeros(size(x));
        u = zeros(size(x));
        %*********************************************************************
        % 4. start iteration
        %*********************************************************************
        tStart = tic;
        for gen = 2:maxgen
            % [o-2]
            %*********************************************************************
            % 3. find mutation population
            %*********************************************************************
            for i = 1:NP
                r = ceil(rand(1,3)*NP);
                while r(1)==r(2) || r(2)== r(3) || min(r)==0 || max(r)>NP
                    r = ceil(rand(1,3)*NP);
                end
                v(i,:) = x(r(1),:) + F*(x(r(2),:) - x(r(3),:));
                for j = 1:D
                    if rand > CR
                        u(i,j) = x(i,j);
                    else
                        u(i,j) = v(i,j);
                    end
                end
                u(i,:)= round(u(i,:));
                u(i,:)=sort(u(i,:));
            end
            for i = 1:NP
                for jj = 1:D
                    u(i,jj) = max(u(i,jj), range_min(jj));
                    u(i,jj) = min(u(i,jj), range_max(jj));
                end
                u(i,:)=sort(u(i,:));
                fitness_child(i,1) = ultrafuzziness([0 0 u(i,:) 255 255],p,alpha);
            end
            for i = 1:NP
                if fitness_parent(i) < fitness_child(i)
                    fitness_parent(i) = fitness_child(i);
                    x(i,:) = u(i,:);
                end
            end
            [globalbest,globalbest_index] = max(fitness_parent);
            global_xbest = sort(x(globalbest_index,:));
            % clc
            % runn
            % fprintf('Optimisation through Differential Evolution\n')
            % fprintf('Generation: %0.5g\nGlobalbest: %2.7g\n', gen, globalbest)
            % fprintf('Best particle position : %0.11g\n', global_xbest)
            %
            globalbest1 = [globalbest1, globalbest];
        end
        tElapsed = toc(tStart);
        % tElapsed
        globalbest1 = [globalbest1, globalbest];
        statistics_f = [statistics_f, globalbest];
        statistics_x = [statistics_x; (global_xbest)];
    end
    % plot(1:NP:NP*50,globalbest1(1:50),'-bs','MarkerFaceColor','b');
    % hold on
    f_mean = mean(statistics_f);
    f_stddev = std(statistics_f);
    best_fitness = max(statistics_f);
    worst_fitness = min(statistics_f);
    x_median = median(statistics_x);
    %% select the threshold points
    % T = [min_val round(x_median) max_val];
    % Thres=T(2:2:2*Th)
    t1=x_median(1:2:D);
    t2=x_median(2:2:D);
    Thres=round((t1+t2)/2);
    X=grayslice(I_bw,[Thres]);
    X1=uint8(255*mat2gray(X));
    timestop=toc(tstart);
    function [Thres,X1] = Thresholding(I1,Th)
    %figure, imshow(I1)
    I=I1(:,:,1:3);
    I_red=I(:,:,1);
    if length(size(I))==3
        I_bw = rgb2gray(I);
    end
    [M, N, O] =size(I);

    L = 256;
    % H=imhist(I_bw);
    H=imhist(I(:,:,2));
    %figure,imhist(I)
    p = H / (M * N);
    %figure(2); plot(h)
    max_val=double(max(max(I_red)));
    min_val=double(min(min(I_red))+1);
    %*********************************************************************
    % 1. Define problem hyperspace and plot in 2D
    %*********************************************************************
    %Th =7;
    % No of thresholds
    D = Th*2; % no of dimensions
    range_min = min_val*ones(1,D);
    range_max = max_val*ones(1,D);% minimum & maximum range;
    alpha=1.5;
    %*********************************************************************
    % 2. initialize the population
    %*********************************************************************
    NP = 10*D ; % population size
    maxgen =100; % no of generations
    F = 0.5;
    CR = 0.9;
    max_runs = 2;
    globalbest1 = [];
    statistics_f = [];
    statistics_x = [];
    tstart = tic;
    for runn = 1:max_runs
        x=[];
        for i = 1 : NP
            for j = 1:D
                x(i, j) = round(range_min(j) + ((range_max(j)-range_min(j))*(rand)));
            end
            x(i,:)=sort(x(i,:));
            fitness_parent(i,1) = ultrafuzziness([0 0 x(i,:) 255 255],p,alpha);
        end
        v = zeros(size(x));
        u = zeros(size(x));
        %*********************************************************************
        % 4. start iteration
        %*********************************************************************
        tStart = tic;
        for gen = 2:maxgen
            % [o-2]
            %*********************************************************************
            % 3. find mutation population
            %*********************************************************************
            for i = 1:NP
                r = ceil(rand(1,3)*NP);
                while r(1)==r(2) || r(2)== r(3) || min(r)==0 || max(r)>NP
                    r = ceil(rand(1,3)*NP);
                end
                v(i,:) = x(r(1),:) + F*(x(r(2),:) - x(r(3),:));
                for j = 1:D
                    if rand > CR
                        u(i,j) = x(i,j);
                    else
                        u(i,j) = v(i,j);
                    end
                end
                u(i,:)= round(u(i,:));
                u(i,:)=sort(u(i,:));
            end
            for i = 1:NP
                for jj = 1:D
                    u(i,jj) = max(u(i,jj), range_min(jj));
                    u(i,jj) = min(u(i,jj), range_max(jj));
                end
                u(i,:)=sort(u(i,:));
                fitness_child(i,1) = ultrafuzziness([0 0 u(i,:) 255 255],p,alpha);
            end
            for i = 1:NP
                if fitness_parent(i) < fitness_child(i)
                    fitness_parent(i) = fitness_child(i);
                    x(i,:) = u(i,:);
                end
            end
            [globalbest,globalbest_index] = max(fitness_parent);
            global_xbest = sort(x(globalbest_index,:));
            % clc
            % runn
            % fprintf('Optimisation through Differential Evolution\n')
            % fprintf('Generation: %0.5g\nGlobalbest: %2.7g\n', gen, globalbest)
            % fprintf('Best particle position : %0.11g\n', global_xbest)
            %
            globalbest1 = [globalbest1, globalbest];
        end
        tElapsed = toc(tStart);
        % tElapsed
        globalbest1 = [globalbest1, globalbest];
        statistics_f = [statistics_f, globalbest];
        statistics_x = [statistics_x; (global_xbest)];
    end
    % plot(1:NP:NP*50,globalbest1(1:50),'-bs','MarkerFaceColor','b');
    % hold on
    f_mean = mean(statistics_f);
    f_stddev = std(statistics_f);
    best_fitness = max(statistics_f);
    worst_fitness = min(statistics_f);
    x_median = median(statistics_x);
    %% select the threshold points
    % T = [min_val round(x_median) max_val];
    % Thres=T(2:2:2*Th)
    t1=x_median(1:2:D);
    t2=x_median(2:2:D);
    Thres=round((t1+t2)/2);
    X=grayslice(I_bw,[Thres]);
    X1=uint8(255*mat2gray(X));
    timestop=toc(tstart);
end
end

function BW2 = Trimp2(BW,pixl)
    [M,N] = size(BW);
    figure, imshow(BW);
    BW2 = BW;
    for i=1:1:M
        for j=1:1:N
            if BW(i,j)>0
                BW(i,j) = 255;
            end
        end
    end
    %figure,imshow(BW);
    BW1 = BW;
    CC = bwconncomp(BW);
    display(CC);
    numPixels = cellfun(@numel,CC.PixelIdxList);
    display(numPixels);
    % Largest connected compnent
    numPixels_temp = sort(numPixels,'descend');
    %display(numPixels_temp);
    idx = find(numPixels==numPixels_temp(1));
    display(idx);
    %figure, imshow(CC.PixelIdxList{idx})
    BW1(CC.PixelIdxList{idx}) = 0;
    % figure, imshow(BW1);
    BW3 = BW-BW1;
    % figure, imshow(BW3);
    CC = bwconncomp(BW3);
    % Number of pixels in each connected components
    numPixels = cellfun(@numel,CC.PixelIdxList);
    % Largest connected compnent
    [biggest,idx] = max(numPixels);
    S = regionprops(CC,'Centroid'); % Calculate centroids for connected components in the image using regionprops.
    cntr = cat(1, S.Centroid); %Concatenate structure array containing centroids into a single matrix.
    centroid_x=round(cntr(idx,1));
    centroid_y=round(cntr(idx,2));
    % hold on
    % plot(centroid_x,centroid_y, 'b*')
    % hold off
    x_left = [];
    x_right = [];
    y_left = [];
    y_right = [];
    count = 1;
    for i=1:1:M
        for j=1:1:centroid_x
            if BW(i,j)==255
                BW2(i,j:j+pixl) = 0;
                break;
            end
        end
    end
    count = 1;
    y_down = 0;
    for i=1:1:M
        for j=N:-1:centroid_x-1
            if BW(i,j)==255
                BW2(i,j:-1:j-pixl) = 0;
                y_down = i;
                break;
            end
        end
    end
    for i=y_down:-1:y_down-pixl
        BW2(i,:) = 0;
    end
    % figure();plot(x_left,y_left,'b*',x_right,y_right,'r*')
    % figure();imshow(BW2)
end

function f=ultrafuzziness(v,p,alpha)
    % if(length(v)==3)
    % u=trimf(1:256,v);
    % uL=u.^(alpha);
    % uU=u.^(1/alpha);
    % f=sum((uU-uL).*p');
    % else
    % f=ultrafuzziness(v(1:3))*ultrafuzziness(v(3:length(v)));
    % end
    if(length(v)==4)
        v=v+1;
        u=trapmf(1:256,v);
        uL=u.^(alpha);
        uU=u.^(1/alpha);
        y=(uU-uL).*p';
        x=sum(y);
        f=0;
        if x~=0
            for i=v(1):v(4)
                if y(i)~=0
                    f=f+(y(i)/x)*log(y(i)/x);
                end
            end
            f=-f;
        end
    else
        f=ultrafuzziness(v(1:4),p,alpha)+ultrafuzziness(v(3:length(v)),p,alpha);
    end
end
