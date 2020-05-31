% calculates 13 shape descriptor for 2D image%
% Last 4 are    incircle/circumcircle  , FmBB/FmBB90  ,   Fmin/Fmax  , Diameter of particle / Fmax        in order
function shapedes(BW,a,op)

t1 = 1;
AllXandY = zeros(1000,48);
%change accordingly
%Delete pre-existing file
%if exist('Pellet.txt', 'file') disp('********The file Pellet.txt already exists*****\n****Deleting the existing one****'); delete Pellet.txt; end
%if exist('Shapedd.txt', 'file') disp('********The file Shapedd.txt already exists*****\n****Deleting the existing one****'); delete Shapedd.txt; end
hold on
coun1= 1;
coun2= 1;
coun3= 1;
ck = 0;
cm= 1;
ko = 1;
for q = 10
 
    
    numberofpixelspercm2 = (a*a*100)/(op*op) ;          %scale change everytime
    %conversion factor for datascale in circle plot to rad for plotting circle
    scale = sqrt(numberofpixelspercm2);
   % sacle is number of pixel/cm 
   scale = scale/10000 ;% per micrometer
        [m,n] = size(BW);
        BW=BW(1:m,1:n);
        %      figure
        %      imshow(BW)-.
        
        CC = bwconncomp(BW);
        L = labelmatrix(CC);
        
        ll=regionprops(L,'PixelList');
        nobj=length(ll);
        
        ptcles=cell(nobj,1);
        
        for i=1:nobj
            ptcles{i} = ll(i).PixelList;
        end
        
        % Access any particle pixels using ptcles{3,1}. Note use of curly brackets.
        
        for i=1:nobj
            Center{i}= mean(ptcles{i,1});
            xymin{i} = min(ptcles{i,1});
            xymax{i} = max(ptcles{i,1});
        end
        
        for i=1:nobj
            ck = ck+1 ;
            ul=[xymin{1,i}];
            lr=[xymax{1,i}];
            
            bw1 = BW(ul(2):lr(2),ul(1):lr(1));
            bw11 = zeros(m,n);
            for x1=ul(2)+1:lr(2)
                for y1=ul(1)+1:lr(1)
                    bw11(x1,y1) = bw1(x1-ul(2),y1-ul(1));
                end
            end
            % figure
            % imshow(bw11)
            
            pixel_count_pelletBW = sum(sum(bw11));
            AreaBW(i) = pixel_count_pelletBW/numberofpixelspercm2; %in cmsq
            rad(i) = sqrt(AreaBW(i)/pi);
            dia(i) = 2*rad(i);    %Diameter array
            myrad = scale*rad(i);  %myrad is distance in terms of pixels
            %%Edge Detection
            
            Img_ED = edge(bw11,'Sobel',0.1); % 0.1??
            
            [y,x] = find(Img_ED); %edge coordinates
            Edges = [x,y]; % Edge co-ordinates
            
            cc=[Center{1,i}];
            
            Cx(i) = cc(1);%mean(x) % x -co-ordinate of object centroid
            Cy(i) = cc(2);%mean(y) % y -co-ordinate of object centroid (to get centroid in same coordinate)
            
            % Cx(i) = mean(x); % x -co-ordinate of object centroid
            % Cy(i) = mean(y);
            %         figure
            %         plot(x,y,'k.') %x, -y are coordinates of the particle with center Cx, Cy
            hold on;
            sigma(i) = (sum((x - Cx(i)).^2+(y - Cy(i)).^2- myrad^2)^2/length(x))^(0.25)/scale; %Sigma array
            %         plot(Cx(i),Cy(i),'k+')
            %         circle(Cx(i),Cy(i),myrad,scale);
            
            % hold on
            % cc=[Center{1,i}]'
            % plot(cc(1),cc(2),'rx')
            % circle(cc(1),cc(2),myrad,scale);
            
            %         hold off
            daspect([1,1,1]);
                    stats =regionprops( 'table' , BW, 'Eccentricity' , 'MajorAxisLength' , 'MinorAxisLength' , 'Perimeter' , 'Area' , 'Centroid' , 'EquivDiameter','Orientation');

            %Method for ordering edges
            a = atan2(y - Cy(i), x - Cx(i));
            [~, order] = sort(a);
            x2= x(order);
            y2= y(order);
            Siv=[x2,y2];
            
            % ----- incircle ----%
            [sir_cx ,sir_cy,sir_r ] = find_inner_circle(x2,y2); % calls find inner circle which gives centre and radius
            [Center2, Radius2] = minboundcircle(x2,y2); % calls minboundcircle function which gives center and radius of circumcircle
            in_bh(cm:cm,1:6) = [sir_cx sir_cy sir_r Center2 Radius2];
            cm = cm+1;
            inner = sir_r; 
            outer = Radius2;
            shape = inner/outer;  
            factor (i) = sqrt(shape); % value of shape descriptor
         
            
            %%%%%%%%%%%%%%%%%%
            
            
            % ==------------fmm -------%%%%%
            
            
            
             k2 = stats.Centroid(i,:);
       xc = k2(1);
       yc = k2(2);
        x22=x2-xc ;
       y22 = y2-yc;
       
       % cordinates of boundary with respect to centeroid

        
        for degree= 0 : 1 : 179 % rotation in 180 degrees
        
        
        
        x3 = x22*cosd(degree) -y22*sind(degree) ; % rotated coordinates
        y3 = x22*sind(degree) + y22*cosd(degree);
        
               
%         plot(x3,y3);
%         hold on;
%         
%         scatter(0,0);
%         hold on;
        
        xmax = max(x3);
        xmin = min(x3);
        
        ymax = max(y3);
        
        ymin = min(y3);
        
        
        A(1:4,degree+1) = [xmin ; xmax ; ymin ;ymax];
        
        B(1:2,degree+1) = [abs(xmin - xmax) ; abs(ymin -ymax)]; % stores all lengths | minimum and maximum lengths will give Fmin/Fmax a new shape descriptor
        
        C(degree+1) =  abs(xmin - xmax) * abs(ymin -ymax); % stores volume of all | minimum of C is minmum bounding box

        
%         plot(xg,t,xs,t,t,ys,t,yg);
%         hold on;

        
        end
            
            C = C';
            [ minbound, ind ] = min(C); % minimum volume index was saved
            if B(1,ind) < B(2,ind)  % for that index value of shape descriptor is needed|  smalller value/ bigger value
            fmbbdivfmbb90(i) = B( 1,ind) / B(2,ind) ;
            else
            fmbbdivfmbb90(i)= B( 2,ind) / B(1,ind) ;
            end
            
            Fmin = min(min(B));
            Fmax = max(max(B));
            
            FmindivFmax(i) = Fmin / Fmax ; % value of shae descriptor
            
            
            Fmaxx(i) = Fmax ; % one more shape descriptor is Dp/fmax for that Fmax is stored here
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%
            
            
            
            
            
            
            
            
            ss = size(Siv);
            AllXandY(1:ss(1),t1:t1) = Siv(:,1);
            AllXandY(1:ss(1),t1+1:t1+1) = Siv(:,2);
            t1 = t1+2;
            perimeterc = 0 ;
            for o = 1 : ss(1) - 1
                perimeterc = perimeterc + sqrt( (Siv(o,1) - Siv(o+1,1))^2 + (Siv(o,2) - Siv(o+1,2))^2 );
            end
            Peri(coun1,1) = perimeterc;
            coun1 = coun1 +1;
            h=numel(y2);
            
            %change accordingly
            for qbeg=1 : 1
                p=1;
                for j= qbeg:q:h-q
                    xmid(p)=(x2(j)+x2(j+q))/2;
                    ymid(p)=(y2(j)+y2(j+q))/2;
                    m1(p)=(y2(j+q)-y2(j))/(x2(j+q)-x2(j));
                    m2(p)=-1/m1(p);
                    if(m1(p)==Inf||m1(p)==-Inf)
                        xmid(p)=x2(j);
                        D(p)=abs(ymid(p)-Cy(i));
                    elseif (m2(p)==-Inf || m2(p)==Inf)
                        ymid(p)=y2(j);
                        D(p)=abs(xmid(p)-Cx(i));
                    else
                        xx(p)=((Cy(i)-(m1(p)*Cx(i)))-(ymid(p)-(m2(p)*xmid(p))))/(m2(p)-m1(p));
                        yy(p)=m2(p)*xx(p)+(ymid(p)-(m2(p)*xmid(p)));
                        D(p)=abs(sqrt((Cx(i)-xx(p))^2+(Cy(i)-yy(p))^2));
                    end
                    ecc(p)=D(p)/scale;
                    
                    p=p+1;
                end
                
                RollingEccen1(i,qbeg)=mean(ecc);
                RollingEccen2(i,qbeg)=rms(ecc);
                
            end
            RollFricEcc1(i)= mean(RollingEccen1(i,:));
            stdRollFricEcc1(i)= std(RollingEccen1(i,:));
            
            RollFricEcc2(i)= mean(RollingEccen2(i,:));
            stdRollFricEcc2(i)= std(RollingEccen2(i,:));
            
            %
        end
        
        DpdivFmax = stats.EquivDiameter./Fmaxx' ;  % shape descriptor
        filename = ['PelletRollFricEcc',num2str(q),'.txt'];               %change in sinter & pellet
        fileID = fopen(filename,'w');
        for i=1:nobj
            fprintf(fileID,'%f %f %f %f\n',RollFricEcc1(i)/stats.EquivDiameter(i),stdRollFricEcc1(i)/stats.EquivDiameter(i),RollFricEcc2(i)/stats.EquivDiameter(i),stdRollFricEcc2(i)/stats.EquivDiameter(i));
            Peri(coun2,2) = stats.MajorAxisLength(i) ;
            Peri(coun3,3) = stats.MinorAxisLength(i) ;
            coun2 = coun2 +1;
            coun3 = coun3 +1;
        end
        
        fclose(fileID);

        
%         fileID = fopen('Pellet1015.txt','a');                                 %change in sinter & pellet
%         for i=1:nobj
%             fprintf(fileID,'%f %f %f %f %f %f %f\n',stats.Area(i)/scale^2,stats.Perimeter(i)/scale,stats.EquivDiameter(i)/scale,stats.Eccentricity(i),stats.MajorAxisLength(i)/scale,stats.MinorAxisLength(i)/scale,sigma(i)); %Sigma array alreay divided by scale
%         end
        for i=1:nobj
            shapedes1(i)=stats.EquivDiameter(i)/stats.MajorAxisLength(i);
            shapedes2(i)=stats.MinorAxisLength(i)/stats.EquivDiameter(i);
            shapedes5(i)=stats.MinorAxisLength(i)/stats.MajorAxisLength(i);
            shapedes3(i)=(  sqrt( 1-(shapedes5(i))^2 )   );
            shapedes4(i)=pi*(stats.EquivDiameter(i))/stats.Perimeter(i);
            shapedes6(i)=stats.Perimeter(i)/(pi*stats.MajorAxisLength(i));
            shapedes7(i)=pi*stats.MinorAxisLength(i)/stats.Perimeter(i);
            shapedes8(i)=sigma(i)/(stats.EquivDiameter(i)/scale);
            shapedes9(i)=RollFricEcc1(i)/(stats.EquivDiameter(i)/scale);
            
        end
%         fclose(fileID);
        
        fileID = fopen('FAZ_shapedes_1to13.txt','a'); % store all 13 shape descriptors in this file
        
        %change in sinter & pellet
        for i=1:nobj
            fprintf(fileID,'%f %f %f %f %f %f  %f %f %f %f %f %f %f\n',shapedes1(i),shapedes2(i),shapedes3(i),shapedes4(i),shapedes5(i),shapedes6(i),shapedes7(i),shapedes8(i), shapedes9(i) , factor(i) ,   fmbbdivfmbb90(i) , FmindivFmax(i) ,   DpdivFmax(i) );
        end
        
%         for i =  1 : nobj
%         storeall(ko,1:13) = [ shapedes1(i),shapedes2(i),shapedes3(i),shapedes4(i),shapedes5(i),shapedes6(i),shapedes7(i),shapedes8(i), shapedes9(i) , factor(i) ,   fmbbdivfmbb90(i) , FmindivFmax(i) ,   DpdivFmax(i) ] ;
%         ko = ko+1 ; 
%         end
 
        fclose(fileID);
        
    
        %%%%%%%%%%%

         fileID = fopen('FAZ_parameters_1to9.txt','a'); % store all 13 shape descriptors in this file
        
        %change in sinter & pellet
        for i=1:nobj
            fprintf(fileID,'%f %f %f %f %f %f %f %f %f %f\n',stats.EquivDiameter(i)/scale,stats.MajorAxisLength(i)/scale,stats.MinorAxisLength(i)/scale,stats.Perimeter(i)/scale,RollFricEcc1(i),Fmin/scale,Fmax/scale,inner/scale , outer/scale, stats.Orientation);
        end
        
%         for i =  1 : nobj
%         storeall(ko,1:13) = [ shapedes1(i),shapedes2(i),shapedes3(i),shapedes4(i),shapedes5(i),shapedes6(i),shapedes7(i),shapedes8(i), shapedes9(i) , factor(i) ,   fmbbdivfmbb90(i) , FmindivFmax(i) ,   DpdivFmax(i) ] ;
%         ko = ko+1 ; 
%         end
 
        fclose(fileID);

        %%%%%%%%%%%%
        
        
        
        
        
        
        
    % clearvars -except numfiles q jpegFiles
end
fileID = fclose('all');
fileID = fopen('Peri.txt','a');
fprintf(fileID,'%f\n',Peri);

fileID = fclose('all');

%[r,p] = corrcoef(shapedes1',shapedes3');    %correlation coefficient calculation
%[r,p] = corr(shapedes1',shapedes9');        %linear or rank correlation
end