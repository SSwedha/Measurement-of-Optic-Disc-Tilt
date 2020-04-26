%%
clear all
clc
close all
 delete 'FAZ_shapedes_1to13.txt'
 delete 'FAZ_parameters_1to9.txt'
 co = 0;
% 70 pixels per mm
hu = zeros(60,1);
hdia = zeros(60,1) ;  
jpegFiles = dir('1');
numfiles = length(jpegFiles);
sd = ones(60,1);
%sd = [6,3,3,6,6,3,3,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6] ;
%for i = 3 : numfiles

 for i = 3 : numfiles
     
    figure(1)
    disp(i-2);
    filename = jpegFiles(i).name;
    fy(i-2,1) =string( filename ); 
    K = imread(filename);
   I = imread(filename);
   lkp = I;
   svf = size(I);
    for lp = 1 : svf(1)
        for lpo = 1 : svf(2)
          
                if(I(lp,lpo,1)+I(lp,lpo,2)-I(lp,lpo,3)<150)
                    I(lp,lpo,1) = 0;
                    I(lp,lpo,2) = 0;
                    I(lp,lpo,3) = 0;
                end
        end
    end
    
    
    I  = im2double(rgb2gray(I));
    lkp = im2double(rgb2gray(lkp));
  %  I  = imcomplement(I) ;
    [a ,b] = size(I);
    %disp([a,b]) ;

    
    
    
  

    
    %   imshow(I);
    %   input('');
    %disp(size(I));
    I=I(ceil(0.2*a):ceil(0.8*a),ceil(0.2*b):ceil(0.8*b)); 
    lkp = lkp(ceil(0.2*a):ceil(0.8*a),ceil(0.2*b):ceil(0.8*b)); 
    %disp(size(I));
    
          %  imshow(lkp);

    
        
        
   
      
       
        %  figure(1);
                  %  imshow([I,BWu,BWs])
        %          return;
        %BWs = imcomplement(BWs);
  %      title('Binary Gradient Mask')
        %se90 = strel('line',3,90);
        % for di = 0 : 90 : 100

       %  input('')
        % return;
        %end
        % BWsdil = BWs ;
        %  figure(2);
        %  imshow(BWsdil);
        %BWsdil = imcomplement(BWsdil);
      closeBW = I;
        
        closeBW  = bwareaopen(closeBW,100);
        closeBW = imcomplement(closeBW);
        closeBW  = bwareaopen(closeBW,100);
        closeBW = imcomplement(closeBW);

        stats = regionprops('table',closeBW,'EquivDiameter','Area');
        if(mean(mean(closeBW))==0)
            co = co+1;
            
         fileID = fopen('FAZ_shapedes_1to13.txt','a'); % store all 13 shape descriptors in this file
        
            fprintf(fileID,'%f %f %f %f %f %f %f %f %f %f %f %f %f \n',0,0,0,0,0,0,0,0,0,0,0,0,0);
     
        fclose(fileID);
           fileID = fopen('FAZ_parameters_1to9.txt','a'); % store all 13 shape descriptors in this file
      
            fprintf(fileID,'%f %f %f %f %f %f %f %f %f%f \n',0,0,0,0,0,0,0,0,0);
       
        fclose(fileID);
        
        
            disp(i-2)
            continue;
        end
        title(i-2);
        closeBW = bwareaopen(closeBW,max(stats.Area)-1);
          if i ==45
        shapedes(closeBW,a,3);
        else
       shapedes(closeBW,a,6);
        end
        
    
        BW3 = bwboundaries(closeBW);
        s = size(BW3);
    
            for j = 1
            BW4 = BW3{j};
            %    imshow(BW3{j});
%             plot(BW4(:,2) , BW4(:,1));
%             hold on;
            end
%         imshow(closeBW);
        s2  = regionprops(closeBW,'centroid','MajorAxisLength','MinorAxisLength');
        centroids = cat(1, s2.Centroid);
        figure(1)
%         imshow(closeBW)
%         hold on        
        centers = s2.Centroid;
        diameters = mean([s2.MajorAxisLength s2.MinorAxisLength],2);
        radii = diameters/2;
        %disp(radii)
         P2 = circle(centroids(:,2), centroids(:,1),2*radii);
      %   figure(2)
        % P = Snake2D(closeBW,P2,radii) ; 
        P=BW4;
       % K = makeboundary(closeBW,P,K);
       % imshow(K)
%         yu = convhull(P); 
%         P = P(yu,:);
        sp = size(P) ; 
%         Perimeter = perimeter(P,kp,sd,i-2) ; 
%         dia =2*sqrt( (polyarea(P(:,1) , P(:,2)) /kp/kp)*sd(i-2)*sd(i-2)/pi) ; 
%         P(sp(1):sp(1) , 1 : 2) = P(1,:) ;
         plot(P(:,2) , P(:,1));
        hold on;
      %  input('');
%         if a ==736
%             hu(i-2,co) = Perimeter ; 
%             hdia(i-2,co) = dia;
%         else
%             hu(i-2,co) = Perimeter ; 
%             hdia(i-2,co) = dia;
%         end
    end
%      input('');
%     df = strcat(int2str(i-2),'.png');
%     saveas(gcf,df)
 
    ld = load('FAZ_shapedes_1to13.txt') ; 
    lo = load('FAZ_parameters_1to9.txt');
    
    MEAN_LO = mean(lo) ; 
    STD_LO = std(lo) ; 
    
    corrld =corrcoef(ld);
    corrlo = corrcoef(lo);