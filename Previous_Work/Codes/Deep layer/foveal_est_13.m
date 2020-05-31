%% MAIN CODE
%Code determines all parameters except tortuosity
%Please ignore unnecessary comments - They are used to debug for specific
%kind of images
clear all 
clc
close all
 delete 'FAZ_shapedes_1to13.txt'
 delete 'FAZ_parameters_1to9.txt'  % initial conditions

hu = zeros(60,1);
hdia = zeros(60,1);  
jpegFiles = dir('1'); % 1 is the name of directory in which all the images are kept
numfiles = length(jpegFiles); % stores number of images in that folder
vad  = zeros(numfiles-2,1); 
sd = ones(60,1);
%sd = [6,3,3,6,6,3,3,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6] ;
 %for i = 3 : numfiles
 for i = 3 : numfiles % one by one goes to all files
    figure(1) 
               % input('');         
    disp(i-2);
    filename = jpegFiles(i).name;
    fy(i-2,1) =string( filename ); %stores filenames
    I  = im2double(rgb2gray(imread(filename)));
    K = imread(filename);
    K = K(1:412,1:412,1:3);
    I  = imcomplement(I) ;
    [a ,b] = size(I);
    disp([a,b]) ;
    %% DOG filter follows %%
    if a == 736 % if size of image is 736 * 718
        I = I(1:718,1:718);
        
        I_ = I;
        kp  = 718 ; 
        d2ex = 0.5;
        sigma2exc = 0.02*d2ex ;
        dinh = 1.5;
        sigma2inh = dinh*0.25;
        
        no_of_iteration =15;
        for li = 1 : no_of_iteration
            
            for lo = 1 : 718
                for lk = 1 : 718
                    
                    k1  = ((d2ex/(2*3.14))/sigma2exc) *exp( -(lo^2+lk^2)/ (2*sigma2exc)) ;
                    k2 = ((dinh/(2*3.14))/sigma2inh) *exp( -(lo^2+lk^2)/ (2*sigma2inh)) ;
                    I(lo,lk) = I(lo,lk) + I(lo,lk)*(k1 -k2) - 0.05 ;
                    
                end
            end
                        
        end
        
        
        
    else
    
        if a==438 % if size of image is 438*420
        I = I(1:420,1:420); % crop unnecessary part
        else
           I = I(1:412,1:412);
        end
        [a,b] = size(I);
    I_ = I;
        %
        d2ex = 0.5;
        sigma2exc = 0.02*d2ex ;
        dinh = 1.5;
        sigma2inh = dinh*0.25;
        kp = a ;
        no_of_iteration =1;
        for li = 1 : no_of_iteration
            
            for lo = 1 : a
                for lk = 1 : a
                    
                    k1  = ((d2ex/(2*3.14))/sigma2exc) *exp( -(lo^2+lk^2)/ (2*sigma2exc)) ;
                    k2 = ((dinh/(2*3.14))/sigma2inh) *exp( -(lo^2+lk^2)/ (2*sigma2inh)) ;
                    I(lo,lk) = I(lo,lk) + I(lo,lk)*(k1 -k2) - 0.02 ;
                   
                end
            end
%            imwrite(I,strcat(int2str(li),'.png'));
            
        end
        %
    end
    %   imshow(I);
    %   input('');
        Ik = I_;
    [a,b] = size(Ik);
    scale_ = 6/a; % scale is 6mm / pixel corresponding
    disp(size(I));
    I=I(ceil(0.20*a):ceil(0.80*a),ceil(0.20*b):ceil(0.80*b));
    %disp(size(I));

    Ik = imcomplement(Ik);
  %  imshow(imcomplement(I));
%      imwrite(imcomplement(I),'complement&crop.png');
%     return;
    Inew = Ik;
     Ik = Ik(ceil(0.10*a):ceil(0.9*a), ceil(0.1*b):ceil(0.9*b));
   Ik_= im2bw(Ik,0.45);
   % imshow(Ik);
   % input('');
    s_ik = size(Ik_); 
     s_ik = size(Ik);
     vad(i-2) = sum(sum(Ik))/(s_ik(1)*s_ik(2)   ); % vessel avascular density 
    I_ = I_(ceil(0.20*a):ceil(0.80*a),ceil(0.20*b):ceil(0.80*b));
    %imshow(I_);
  %  input('');
    [~,threshold] = edge(I,'Prewitt');
    figure(1) ; 
    title(int2str(i-2));
    hold on;
    co = 1 ;
    I= imcomplement(I) ;
    for ko = 0.43
         abu = size(I) ; 
       au = abu(1);
       bu = abu(2);
        fudgeFactor = ko;
       BWk = I(ceil(au*0.30) :ceil( 0.70*au) , ceil(bu*0.30):ceil(0.70*bu)) ;        
     
       BWu = edge(I,'Prewitt',threshold);
        BWs = edge(I,'Prewitt',threshold * fudgeFactor);
        Bwo = edge(BWk,'Prewitt',threshold*.45 );
       % BWs(ceil(au*0.30) :ceil( 0.70*au) , ceil(bu*0.30):ceil(0.70*bu))  = Bwo; 
      %  imshow([BWs,Bku]);
              [~,threshold2] = edge(Ik,'Prewitt');
        Bu = edge(I,'Prewitt',threshold2-0.15);
       % input('');
       % imshow([Ik,Bu,bwmorph(Bu,'remove')])
        % input('');
        %  figure(1);
        % imshow([Bku,new,BWs]);
       % imshow(Bku);
       
%         imshow(new);
% 
  %imshow(BWs);
%  input('');
% %                  return;
    % input('')
         binary = im2bw(Ik,0.65);
         skeleton = bwmorph(binary, 'skel', inf); % finds skeleton of the image
         vid(i-2) = (sum(sum(binary)) ) / sum(sum(skeleton)) ; 
         vid(i-2) = scale_*vid(i-2); % Stores Vessel Diameter index
        %  figure(1);
%         imshow([imcomplement(I),BWs])
%         input('');
        %          return;
        %BWs = imcomplement(BWs);
  %      title('Binary Gradient Mask')
        %se90 = strel('line',3,90);
        % for di = 0 : 90 : 100
        temp = BWs;
         temp  = bwareaopen(temp,4);
         
%         imshow(temp);
%         input('');
        % imshow(temp);
         %input('');
         for ty =  0 : 45 : 90
            se0 = strel('line',2,ty);
            BWsdil = imdilate(temp,[se0]);
           temp = BWsdil;
         end
        temp =   bwareaopen(temp,200);
        temp = imcomplement(temp);
        temp = bwareaopen(temp,200);
       % imshow(temp);
         %imshow([imcomplement(I),temp,BWs]);
% %         return;
          
         
        % return;
        %end
        % BWsdil = BWs ;    
        %  figure(2);
        %  imshow(BWsdil);
%           imwrite(BWsdil,'Imagedilation.png');
%           return;
        %BWsdil = imcomplement(BWdil);
         se = strel('disk',3);
         closeBW = imclose(temp,se);
       
         se = strel('disk',5,0);
           erodedI  = imerode(temp,se);
          se = strel('disk',6,0);
         erodedI  = imdilate(erodedI,se);
           closeBW = erodedI;
        %   imshow( [erodedI closeBW])
          % input('')
      %  closeBW = temp;
      %  title('Dilated Gradient Mask')
     %   BWfill = bwareaopen(BWsdil,1000);
        % BWdfill = imfill(BWsdil,'holes');
        %   figure(3);
%          imshow(closeBW)
%          input('');
        %dilatedImage = imdilate(imcomplement(BWfill),strel('disk',6));
        %imshow(dilatedImage);
        % % figure(4)
        % closeBW = imcomplement(closeBW);
                 %imshow(closeBW);
        closeBW  = bwareaopen(closeBW,500);
        closeBW = imcomplement(closeBW);
        closeBW  = bwareaopen(closeBW,500);
        %closeBW = imcomplement(closeBW);
        
                % imshow(imcomplement(closeBW));
        %         return;
       % input('');
        stats = regionprops('table',closeBW,'EquivDiameter','Area');
        if(mean(mean(closeBW))==0)
            co = co+1;
                  
         fileID = fopen('FAZ_shapedes_1to13.txt','a'); % store all 13 shape descriptors in this file
        
            fprintf(fileID,'%f %f %f %f %f %f %f %f %f %f %f %f %f \n',0,0,0,0,0,0,0,0,0,0,0,0,0);
     
        fclose(fileID);
           fileID = fopen('FAZ_parameters_1to9.txt','a'); % store all 10 shape descriptors in this file
      
            fprintf(fileID,'%f %f %f %f %f %f %f %f %f %f \n',0,0,0,0,0,0,0,0,0);
       
        fclose(fileID);
            continue;
        end
        
        closeBW = bwareaopen(closeBW,max(stats.Area)-1);
          %%%% vascular density
   % imshow([imcomplement(I_) imcomplement(I_.*closeBW)])
    sclo = sum(sum(closeBW)) ;
    newr = I_.*closeBW ; 
  %  imshow(newr);
    snewr  = sum(sum(newr)) ;
    percent(i) = 100*(1-(snewr/sclo));
    %%%%%%
    
      %  closeBW = removeb(closeBW);
    closeBW = imcomplement(closeBW);
        BW3 = bwboundaries(closeBW);
        s = size(BW3);
    
            for j = 1 : s(1)
            BW4 = BW3{j};
            %    imshow(BW3{j});
%             plot(BW4(:,2) , BW4(:,1));
%             hold on;
            end
%         imshow(closeBW);
        s2  = regionprops(closeBW,'centroid','MajorAxisLength','MinorAxisLength');
        centroids = cat(1, s2.Centroid);
     
%         imshow(closeBW)
%         hold on        
        centers = s2.Centroid;
        diameters = mean([s2.MajorAxisLength s2.MinorAxisLength],2);
        radii = diameters/2;
        disp(radii)
         P2 = circle(centroids(:,2), centroids(:,1),2*radii);
      %   figure(2)
%          P = Snake2D(temp,P2,radii) ; 
%          P = ceil(P);
         P=BW4;
%           yu = convhull(P); 
% % %         
%           P = P(yu,:);
        sp = size(P) ;
        windowWidth = 17;
polynomialOrder = 2;
P(:,1) = sgolayfilt(P(:,1), polynomialOrder, windowWidth);
P(:,2) = sgolayfilt(P(:,2), polynomialOrder, windowWidth);
        %imshow(closeBW);
        Op = K;
        K = makeboundary(closeBW,P,K);
      %  if any image is of other dimension please for that specify it
        if i ==2 || i == 10
        shapedes(closeBW,412,3);
        else
       shapedes(closeBW,412,6);
        end
        subplot(1,3,1);
        
        imshow(Op);
       title('Clear Image');
        subplot(1,3,2);
        
        imshow(K);
        title('Automated');
        subplot(1,3,3);
       
        imshow(imread(  strcat('D:\Intern UW\testing_data_deep_layer\Arpit_SL_DL_2020\deep layer\2\'  , int2str(i-2) , '.jpg')  ));
         title('Manual');
        %imshow([Op ,K]);
  %input('');
% P(:,1) = P(:,1)+ 0.32*438 ;
% P(:,2) = P(:,2)+ 0.32*420 ;
    end
    % input('');
     df = strcat(int2str(i-2),'.png');
     saveas(gcf,df)
     
 end
    ld = load('FAZ_shapedes_1to13.txt') ; 
    lo = load('FAZ_parameters_1to9.txt');
    vid = vid';
    diame = lo(:,1);
    
    area = pi*(diame.^2)/4 ;
    All = [area lo vad vid]; % all parameters except tortuosity
    MEAN_LO = mean(lo) ; 
    STD_LO = std(lo) ; 
    corrld =corrcoef(ld);
    corrlo = corrcoef(lo);
    % disp("mean vad ");
disp(mean(vad));
%disp("mean vid");
disp(mean(vid));