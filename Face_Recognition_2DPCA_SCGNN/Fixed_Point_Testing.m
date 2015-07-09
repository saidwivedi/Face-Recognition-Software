function [opti_parameter,fixed_accu,fixed_approx_accu,temp_accu]= Fixed_Point_Testing(b,iw,lw,test_input,test_target,optimal_hidden_neuron,num_subjects,TsNum,max_accu_fi,float_accu,max_accu)
     
     word_range = [8];
     fraction_range = [6];
     for prec = 1:length(word_range)
         word_length = word_range(prec);
         fraction_length = fraction_range(prec);
         weight_input_fi = fi(iw{1,1},1,word_length,fraction_length);     
         weight_hidden_fi = fi(lw{2,1},1,word_length,fraction_length);    
         bias_input_fi = fi(b{1,1},1,word_length,fraction_length);        
         bias_hidden_fi = fi(b{2,1},1,word_length,fraction_length);       


         input_temp = removeconstantrows(mapminmax(test_input));
         test_input_fi = fi(input_temp,1,word_length,fraction_length);   

         weighted_sum_1_fi = fi([],1,word_length,fraction_length);   
         weighted_sum_2_fi = fi([],1,word_length,fraction_length);  
         hidden_fi = fi([],1,word_length,fraction_length);           
         output_fi = fi([],1,word_length,fraction_length);  
         hidden_fi_approx = fi([],1,word_length,fraction_length);           
         output_fi_approx = fi([],1,word_length,fraction_length); 

         for j = 1:num_subjects*TsNum
            %%%%%% 1st Layer Calculation %%%%%

             for k = 1:optimal_hidden_neuron
                weighted_sum_1_fi = sum(times(test_input_fi(:,j),weight_input_fi(k,:)'));
                hidden_fi(k,j) = 2/(1+exp(-2*(double(weighted_sum_1_fi) + double(bias_input_fi(k)))))-1;
                hidden_fi_approx(k,j) = tansig_approx(double(weighted_sum_1_fi) + double(bias_input_fi(k)));
             end

            %%%%%% 2nd Layer Calculation %%%%%
             for k = 1:num_subjects
                weighted_sum_2_fi = sum(times(hidden_fi(:,j),weight_hidden_fi(k,:)'));
                output_fi(k,j) = 2/(1+exp(-2*(double(weighted_sum_2_fi) + double(bias_hidden_fi(k)))))-1;
                output_fi_approx(k,j) = tansig_approx(double(weighted_sum_2_fi) + double(bias_hidden_fi(k)));
             end

         end

         [fixed_accu,cm_fi] = confusion(test_target,double(output_fi));
         [fixed_approx_accu,cm_fi_approx] = confusion(test_target,double(output_fi_approx));
         opti_parameter = [];

         if (100*(1-fixed_accu) > max_accu_fi) && (100*(1-fixed_accu) == 100*(1-float_accu)) && (100*(1-float_accu) == max_accu) && ((100*(1-fixed_approx_accu))==100*(1-fixed_accu))
             opti_parameter.max_accu_fi = 100*(1-fixed_accu);
             opti_parameter.opti_weight_input_fi = weight_input_fi;
             opti_parameter.opti_weight_hidden_fi = weight_hidden_fi;
             opti_parameter.opti_bias_input_fi = bias_input_fi;
             opti_parameter.opti_bias_hidden_fi = bias_hidden_fi;
             opti_parameter.opti_test_input_fi = test_input_fi;
             opti_parameter.opti_weighted_sum_1_fi = weighted_sum_1_fi;
             opti_parameter.opti_weighted_sum_2_fi = weighted_sum_2_fi;
             opti_parameter.opti_hidden_fi = hidden_fi;
             opti_parameter.opti_output_fi = output_fi;
             opti_parameter.opti_test_target = test_target;
             opti_parameter.opti_hidden_neuron = optimal_hidden_neuron;
         end

         fprintf('Accuracy with Fixed Point\n');
         fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-fixed_accu));
         fprintf('Accuracy with Fixed Point with Approximation\n');
         fprintf('Percentage Correct Classification   : %f%%\n\n', 100*(1-fixed_approx_accu));
         temp_accu(prec) = 100*(1-fixed_accu);  
     end
end