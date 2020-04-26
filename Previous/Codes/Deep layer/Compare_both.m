% compare both
deep_layer_syst = load('D:\Intern UW\testing_data_deep_layer\Arpit_SL_DL_2020\deep layer\FAZ_parameters_1to9.txt');
deep_layer_manual = load('D:\Intern UW\testing_data_deep_layer\Arpit_SL_DL_2020\deep_layer_manual\FAZ_parameters_1to9.txt');
super_auto = load('D:\Intern UW\testing_data_deep_layer\Arpit_SL_DL_2020\Arpit_SL_DL_2020\FAZ_parameters_1to9.txt');

% 
% super_auto = super_auto(1:11,2:3);
% deep_layer_manual = deep_layer_manual(1:11,2:3);
% Error2 = 100* abs ((super_auto - deep_layer_manual )./deep_layer_manual );
% Error2 = Error2(1:11,2:3);
% Error2  = Error2';
% Error2 = mean(Error2);
% K_2(1) = Error2(10);
% K_2(2) = Error2(2);
% %K_ = mean(K_);
% Error2 = sum(Error2) - sum(K_2) ;
% Error2 = Error2/9;
% 
Error = 100* abs ((deep_layer_syst - deep_layer_manual )./deep_layer_manual );
Error = Error(1:11,1:9);
Error  = Error';
Error = mean(Error);
disp(mean(Error));
K_(1) = Error(10);
K_(2) = Error(2);
%K_ = mean(K_);
Error = sum(Error) - sum(K_) ;
Error = Error/9;
