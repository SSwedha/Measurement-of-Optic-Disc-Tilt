A=load('Pellet8to10_shape_1_to_13.txt');
[rhop , pvalp] = corr(A,'Type','Pearson') ;
[rhok , pvalk] = corr(A,'Type','Kendall');
[rhos , pvals] = corr(A,'Type','Spearman');
str = '_P_8_to_10';
  fileID = fopen(strcat('rhop',str,'.txt'),'a'); % store all 13 shape descriptors in this file
        for i=1 : 13
                for j =  1 : 13
            fprintf(fileID,'%f ', rhop(i,j) );
                end
                 fprintf(fileID,'\n ');
        end
   fileID = fopen(strcat('rhok',str,'.txt'),'a'); % store all 13 shape descriptors in this file
        for i=1 : 13
                for j =  1 : 13
            fprintf(fileID,'%f ', rhok(i,j) );
                end
                 fprintf(fileID,'\n ');
        end
       
          fileID = fopen(strcat('rhos',str,'.txt'),'a'); % store all 13 shape descriptors in this file
        for i=1 : 13
                for j =  1 : 13
            fprintf(fileID,'%f ', rhos(i,j) );
                end
                 fprintf(fileID,'\n ');
        end
          fileID = fopen(strcat('pvalp',str,'.txt'),'a'); % store all 13 shape descriptors in this file
        for i=1 : 13
                for j =  1 : 13
            fprintf(fileID,'%f ', pvalp(i,j) );
                end
                 fprintf(fileID,'\n ');
        end
          fileID = fopen(strcat('pvalk',str,'.txt'),'a'); % store all 13 shape descriptors in this file
        for i=1 : 13
                for j =  1 : 13
            fprintf(fileID,'%f ', pvalk(i,j) );
                end
                 fprintf(fileID,'\n ');
        end
          fileID = fopen(strcat('pvals',str,'.txt'),'a'); % store all 13 shape descriptors in this file
        for i=1 : 13
                for j =  1 : 13
            fprintf(fileID,'%f ', pvals(i,j) );
                end
                 fprintf(fileID,'\n ');
        end
        fclose('all');