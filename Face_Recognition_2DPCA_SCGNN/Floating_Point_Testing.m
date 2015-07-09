function float_accu = Floating_Point_Testing(b,iw,lw,test_input,test_target,optimal_hidden_neuron,num_subjects,TsNum)

     weight_input = iw{1,1};
     weight_hidden = lw{2,1};
     bias_input = b{1,1};
     bias_hidden = b{2,1};

     input_temp = mapminmax(test_input);
     test_input_cu = removeconstantrows(input_temp);


     hidden = [];
     %hidden_approx = [];
     output = [];
     %output_approx = [];
     indx = [];
     for j = 1:num_subjects*TsNum
        %%%%%% 1st Layer Calculation %%%%%

         for k = 1:optimal_hidden_neuron
            weighted_sum_1 = sum(times(test_input_cu(:,j),weight_input(k,:)'));
            hidden(k,j) = 2/(1+exp(-2*(weighted_sum_1 + bias_input(k))))-1;
            %hidden_approx(k,j) = tansig_approx(weighted_sum_1 + bias_input(k));
         end

        %%%%%% 2nd Layer Calculation %%%%%
         for k = 1:num_subjects
            weighted_sum_2 = sum(times(hidden(:,j),weight_hidden(k,:)'));
            output(k,j) = 2/(1+exp(-2*(weighted_sum_2 + bias_hidden(k))))-1;
            %output_approx(k,j) = tansig_approx(weighted_sum_2 + bias_hidden(k));
         end

     end

     [float_accu,cm_cu] = confusion(test_target,output);
     %[c_cu_approx,cm_cu_approx] = confusion(test_target,output_approx);
     fprintf('Accuracy with Floating Point\n');
     fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-float_accu));  
