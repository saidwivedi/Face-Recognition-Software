function target = Target_Calculation(num_subjects,TrNum,TsNum)
    
    train_target = zeros(num_subjects,TrNum*num_subjects);
    test_target = zeros(num_subjects,TsNum*num_subjects);
    tr_count = 0; 
    test_count =0;
    for i = 1:num_subjects
        for j = 1:TrNum
            tr_count = tr_count +1;
            train_target(i,tr_count) = 1;
        end
        for j = 1:TsNum
            test_count = test_count+1;
            test_target(i,test_count) = 1;
        end 
    end
    target = [train_target test_target];

end