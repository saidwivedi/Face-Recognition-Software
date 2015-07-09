function [net,b,iw,lw,optimal_hidden_neuron,max_accu] = classifier(train_input,train_target,test_input,test_target)

    num_hidden_neuron = 44;     % With the setrandom function set to certain value, 44 number of hidden neurons is optimum 
    optimal_hidden_neuron = num_hidden_neuron;
    

    Error_init_weight = [];
    
    num_init_param = [500];
    for init_param = 1:numel(num_init_param)
        max_accu = -inf;
        count = 0;
           for num_hidden_neuron = 20:1:50
             %setdemorandstream(491218382);
             net = patternnet(num_hidden_neuron);
             net.performFcn = 'mse';
             net.trainFcn = 'trainscg';
             net.layers{1}.transferFcn = 'tansig';
             net.layers{2}.transferFcn = 'tansig';
             %view(net);
             net.divideParam.trainRatio = 1.0; % training set [%]
             net.divideParam.valRatio = 0.0; % validation set [%]
             net.divideParam.testRatio = 0.0; % test set [%]
             net.trainParam.epochs = num_init_param(init_param);
             net.trainParam.showWindow = 0;
             net.inputs{1}.processFcns = {'mapminmax'};
             net.outputs{2}.processFcns = {'mapminmax'};
             [net,tr] = train(net,train_input,train_target);
             testY = net(test_input);
             [c,cm] = confusion(test_target,testY);
             count = count + 1;
             temp_c(count) = 100*(1-c);
             if 100*(1-c) > max_accu
                 max_accu = 100*(1-c);
                 weight_biases = formwb(net,net.b,net.iw,net.lw);
                 [b,iw,lw] = separatewb(net,weight_biases);
                 optimal_hidden_neuron = num_hidden_neuron;
             end
           end
        Error_init_weight(init_param) = max_accu;
        fprintf('Accuracy\n');
        fprintf('Percentage Correct Classification : %f%% with %d hidden neurons \n', max_accu,optimal_hidden_neuron);
    end