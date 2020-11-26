%%%backup

%%
clear all
clc
close all
 delete 'FAZ_shapedes_1to13.txt'
 delete 'FAZ_parameters_1to9.txt'
 
% 70 pixels per mm

hu = zeros(60,1);
hdia = zeros(60,1) ;  
jpegFiles = dir('1');

numfiles = length(jpegFiles);
vad  = zeros(numfiles-2,1);
sd = ones(60,1);
%sd = [6,3,3,6,6,3,3,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6] ;
 %for i = 3 : numfiles
 for i = 3 : numfiles
    figure(1) 
         %       input('');         
    disp(i-2);
    filename = jpegFiles(i).name;
   % fy(i-2,1) =string( filename ); 
    I  = im2double(rgb2gray(imread(filename)));
    K = imread(filename);
    I  = imcomplement(I) ;
    [a ,b] = size(I);
    disp([a,b]) ;
    
    if a == 736
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
    
        if a==438
        I = I(1:420,1:420);
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
        no_of_iteration =25;
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
    scale_ = 6/a;
    disp(size(I));
    I=I(ceil(0.32*a):ceil(0.68*a),ceil(0.32*b):ceil(0.68*b)); 
    disp(size(I));

    Ik = imcomplement(Ik);
    Inew = Ik;
     Ik = Ik(ceil(0.10*a):ceil(0.9*a), ceil(0.1*b):ceil(0.9*b));
   Ik_= im2bw(Ik,0.45);
  %  imshow(Ik);
    s_ik = size(Ik_);
   
 
     s_ik = size(Ik);
     vad(i-2) = sum(sum(Ik))/(s_ik(1)*s_ik(2)   );
     
    I_ = I_(ceil(0.32*a):ceil(0.68*a),ceil(0.32*b):ceil(0.68*b));
%        imshow(imcomplement(I_)) ; 
%     
%     
%    return;
    [~,threshold] = edge(I,'Prewitt');
    figure(1) ; 
    title(int2str(i-2));
    hold on;
    co = 1 ;
    for ko = 0.45
         abu = size(I) ; 
       au = abu(1);
       bu = abu(2);
        fudgeFactor = ko;
        BWk = I(ceil(au*0.25) :ceil( 0.75*au) , ceil(bu*0.25):ceil(0.75*bu)) ; 
        
        
        BWu = edge(I,'Prewitt',threshold);
        new = edge(I,'Prewitt',threshold*0.35);
        BWs = edge(I,'Prewitt',threshold * fudgeFactor);
        Bku = BWs;
        Bwo = edge(BWk,'Prewitt',threshold*.45);
        BWs(ceil(au*0.25) :ceil( 0.75*au) , ceil(bu*0.25):ceil(0.75*bu))  = Bwo; 
      
              [~,threshold2] = edge(Ik,'Prewitt');
        Bu = edge(Ik,'Prewitt',threshold2-0.15);
         %imshow([Ik,Bu,bwmorph(Bu,'remove')])
        % input('');
        %  figure(1);
   imshow([Bku,new,BWs]);
%   return;
      %  imshow(Bku);
       
%         imshow(new);
% 
%        % imshow(BWs);
                 % return;
%    input('')
         vid(i-2) = sum(sum(Ik_)) / sum(sum(Bu)) ; 
         vid(i-2) = scale_*vid(i-2) * 1000;
        %  figure(1);
%         imshow([imcomplement(I),BWs])
%         input('');
        %          return;
        %BWs = imcomplement(BWs);
  %      title('Binary Gradient Mask')
        %se90 = strel('line',3,90);
        % for di = 0 : 90 : 100
        temp = BWs;
      %  imshow(temp);
       % temp  = bwareaopen(temp,5);
        imshow(temp)
        %return;
         for ty =  0  : 45 : 180
            se0 = strel('line',3,ty);
            BWsdil = imdilate(temp,[se0]);
           % temp = BWsdil;
         end
        imshow([temp,BWsdil]);
     %   return;
%        imshow([imcomplement(I),temp,BWs]);
%         return;
       %  input('')
        % return;
        %end
        % BWsdil = BWs ;    
        %  figure(2);
   
        %BWsdil = imcomplement(BWdil);
        se = strel('disk',5);
        closeBW = imclose(BWsdil,se);
         imshow([BWsdil,closeBW]);
       %  return;
         %       imshow(BWsdil);
          %return;
%         input('');
      %  title('Dilated Gradient Mask')
        BWfill = bwareaopen(BWsdil,1000);
        % BWdfill = imfill(BWsdil,'holes');
        %   figure(3);
        %   imshow(BWfill)
        dilatedImage = imdilate(imcomplement(BWfill),strel('disk',6));
        
        % % figure(4)
        % closeBW = imcomplement(closeBW);
        %         imshow(closeBW);
        closeBW  = bwareaopen(closeBW,500);
        closeBW = imcomplement(closeBW);
        closeBW  = bwareaopen(closeBW,500);
        %closeBW = imcomplement(closeBW);
        imshow(closeBW)
      %  return;
                 % imshow(imcomplement(closeBW));
        %         return;
        %
        stats = regionprops('table',closeBW,'EquivDiameter','Area');
        if(mean(mean(closeBW))==0)
            co = co+1;
                  
         fileID = fopen('FAZ_shapedes_1to13.txt','a'); % store all 13 shape descriptors in this file
        
            fprintf(fileID,'%f %f %f %f %f %f %f %f %f %f %f %f %f \n',0,0,0,0,0,0,0,0,0,0,0,0,0);
     
        fclose(fileID);
           fileID = fopen('FAZ_parameters_1to9.txt','a'); % store all 13 shape descriptors in this file
      
            fprintf(fileID,'%f %f %f %f %f %f %f %f %f \n',0,0,0,0,0,0,0,0,0);
       
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
%          yu = convhull(P); 
% %         
%          P = P(yu,:);
%         sp = size(P) ;
        K = makeboundary(closeBW,P,K);
        if i ==45
        shapedes(closeBW,a,3);
        else
       shapedes(closeBW,a,6);
        end
        figure(2)
       imshow(K);
        hold on ;
       % close all;
%        figure()
%         imshow([closeBW, new_BW])
%         Perimeter = perimeter(P,kp,sd,i-2) ; 
%         dia =2*sqrt( (polyarea(P(:,1) , P(:,2)) /kp/kp)*sd(i-2)*sd(i-2)/pi) ; 
P(:,1) = P(:,1)+ 0.32*400;
P(:,2) = P(:,2)+ 0.32*412 ;
        
%         P(sp(1):sp(1) , 1 : 2) = P(1,:) ;
%         plot(P(:,2) , P(:,1));
%         hold on;
      %  input('');
%         if a ==736
%             hu(i-2,co) = Perimeter ; 
%             hdia(i-2,co) = dia;
%         else
%             hu(i-2,co) = Perimeter ; 
%             hdia(i-2,co) = dia;
%         end
    end
    % input('');
     df = strcat(int2str(i-2),'.png');
     saveas(gcf,df)
     
 end
    ld = load('FAZ_shapedes_1to13.txt') ; 
    lo = load('FAZ_parameters_1to9.txt');
    
    MEAN_LO = mean(lo) ; 
    STD_LO = std(lo) ; 
    
    corrld =corrcoef(ld);
    corrlo = corrcoef(lo);
    % disp("mean vad ");
disp(mean(vad));
%disp("mean vid");
disp(mean(vid));