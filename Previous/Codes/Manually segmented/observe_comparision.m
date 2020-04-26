
clear all

for i = 1 : 50
   
 cd 'D:\Intern UW\faz from shankar nethralaya\faz-20190602T053900Z-001\faz\clear'   
 I1  =    imread(strcat(int2str(i),'.png'));
 
 cd 'D:\Intern UW\faz from shankar nethralaya\faz-20190602T053900Z-001\faz\manual'
 I2  =    imread(strcat((int2str(i)),'.png'));
 
 cd 'D:\Intern UW\faz from shankar nethralaya\faz-20190602T053900Z-001\faz\syst'
 I3  =    imread(strcat((int2str(i)),'.png'));
    
 imshow([I1,I2,I3]) ; 
    input('');
end