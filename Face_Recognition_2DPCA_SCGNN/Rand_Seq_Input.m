function [train_input,train_target,test_input,test_target] = Rand_Seq_Input(input,target,sample_set,rand_seq,num_subjects,TrNum)

    
    input = input(:,rand_seq);
    
    target = target(:,rand_seq);
    
    train_input = input(:,1:TrNum*num_subjects);
    test_input = input(:,(TrNum*num_subjects)+1:sample_set*num_subjects);
    
    train_target = target(:,1:TrNum*num_subjects);
    test_target = target(:,(TrNum*num_subjects)+1:sample_set*num_subjects);