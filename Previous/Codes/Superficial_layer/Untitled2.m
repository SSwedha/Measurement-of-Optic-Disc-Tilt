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
sd = ones(60,1);
%sd = [6,3,3,6,6,3,3,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6] ;

for i = 3 : numfiles
    figure(1) 
    %      if(i==9)
    %          co ntinue;
    %      end
    disp(i-2);
    filename = jpegFiles(i).name;
    fy(i-2,1) =string( filename ); 
    I  = im2double(rgb2gray(imread(filename)));
    I  = imcomplement(I) ;
    [a ,b] = size(I);
    disp([a,b]) ;
% end
% %%
% return;
% for i = 1 : 10
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
            
       %     imwrite(I,strcat(no_of_iteration,'.jpg'));
            
        end
        
        
        
    else
        I = I(1:420,1:420);
        I_ = I;
        %%
        d2ex = 0.5;
        sigma2exc = 0.02*d2ex ;
        dinh = 1.5;
        sigma2inh = dinh*0.25;
        kp = 420 ;
        no_of_iteration =15;
        for li = 1 : no_of_iteration
            
            for lo = 1 : 420
                for lk = 1 : 420
                    
                    k1  = ((d2ex/(2*3.14))/sigma2exc) *exp( -(lo^2+lk^2)/ (2*sigma2exc)) ;
                    k2 = ((dinh/(2*3.14))/sigma2inh) *exp( -(lo^2+lk^2)/ (2*sigma2inh)) ;
                    I(lo,lk) = I(lo,lk) + I(lo,lk)*(k1 -k2) - 0.02 ;
                    
                end
            end
       %     imwrite(I,strcat(int2str(li),'.png'));
            
        end
        %%
    end
  
    
    
    %   imshow(I);
    %   input('');
    disp(size(I));
    I=I(ceil(0.2*a):ceil(0.8*a),ceil(0.2*b):ceil(0.8*b)); 
    disp(size(I));
    
    I_ = I_(ceil(0.2*a):ceil(0.8*a),ceil(0.2*b):ceil(0.8*b));
    
    [~,threshold] = edge(I,'Prewitt');
    figure(1) ; 
   % imshow(imcomplement(I_));
    hold on;
    co = 1 ;
    for ko = 0.58
        
        fudgeFactor = ko;
        BWu = edge(I,'Prewitt',threshold);
        BWs = edge(I,'Prewitt',threshold * fudgeFactor);
        %  figure(1);
        %            imshow([I,BWu,BWs])
        %          return;
        %BWs = imcomplement(BWs);
  %      title('Binary Gradient Mask')
        %se90 = strel('line',3,90);
        % for di = 0 : 90 : 100
        temp = BWs;
        for ty =  0 : 45 : 90
            se0 = strel('line',3,ty);
            BWsdil = imdilate(temp,[se0]);
            temp = BWsdil;
        end
        % imshow([BWs,BWsdil]);
        % return;
        %end
        % BWsdil = BWs ;
        %  figure(2);
        %  imshow(BWsdil);
        %BWsdil = imcomplement(BWsdil);
        se = strel('disk',7);
        closeBW = imclose(BWsdil,se);
        
      %  title('Dilated Gradient Mask')
        BWfill = bwareaopen(BWsdil,1000);
        % BWdfill = imfill(BWsdil,'holes');
        %   figure(3);
        %   imshow(BWfill)
        dilatedImage = imdilate(imcomplement(BWfill),strel('disk',5));
        % % figure(4)
        % closeBW = imcomplement(closeBW);
        %         imshow(closeBW);
        closeBW  = bwareaopen(closeBW,500);
        closeBW = imcomplement(closeBW);
        closeBW  = bwareaopen(closeBW,500);
        %closeBW = imcomplement(closeBW);
        
        %          imshow(closeBW);
        %         return;
        %
        stats = regionprops('table',closeBW,'EquivDiameter','Area');
        if(mean(mean(closeBW))==0)
            co = co+1;
            continue;
        end
        
        closeBW = bwareaopen(closeBW,max(stats.Area)-1);
          %%%% vascular density
   % imshow([imcomplement(I_) imcomplement(I_.*closeBW)])
    sclo = sum(sum(closeBW)) ;
    newr = I_.*closeBW ; 
    imshow(newr);
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
        figure(1)
%         imshow(closeBW)
%         hold on        
        centers = s2.Centroid;
        diameters = mean([s2.MajorAxisLength s2.MinorAxisLength],2);
        radii = diameters/2;
%         P2 = circle(centroids(:,2), centroids(:,1),3*radii);
%         figure(2)
%         P = Snake2D(closeBW,P2) ; 
        P=BW4;
        sp = size(P) ; 
        Perimeter = perimeter(P,kp,sd,i-2) ; 
        dia =2*sqrt( (polyarea(P(:,1) , P(:,2)) /kp/kp)*sd(i-2)*sd(i-2)/pi) ; 
        P(sp(1):sp(1) , 1 : 2) = P(1,:) ;
        plot(P(:,2) , P(:,1));
        hold on;

%         if a ==736
%             hu(i-2,co) = Perimeter ; 
%             hdia(i-2,co) = dia;
%         else
%             hu(i-2,co) = Perimeter ; 
%             hdia(i-2,co) = dia;
%         end
%         co = co+1;
    end
        
    df = strcat(int2str(i-2),'.png');
    saveas(gcf,df)
end
% imshow(dilatedImage);
% title('Binary Image with Filled Holes')