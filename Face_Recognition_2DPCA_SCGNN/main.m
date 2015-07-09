clc
clear all
close all

%% Display format for Fixed Point Representation
    
%     originalFormat = get(0, 'format');
%     format loose
%     format long g
%     fiprefAtStartOfThisExample = get(fipref);
%     reset(fipref);

%% Adding Path, Selecting Dimension and other initialisation
    
    addpath('C:\Users\saidwivedi\Github\Face-Recognition-Software\Face_Recognition_2DPCA_SCGNN');
    Results=[];
    num_subjects=40; % number of subjects
    total_sample = 10;
    dimensions=2; % Number of desired dimensions
    DatabasePathORL='C:\Users\saidwivedi\Github\Face-Recognition-Software\Face_Recognition_2DPCA_SCGNN\Database\ORL\s'; % DatabasePathORL
    DatabasePathUMIST = 'C:\Users\saidwivedi\Github\Face-Recognition-Software\Face_Recognition_2DPCA_SCGNN\Database\UMIST\s';
    ff='.pgm'; % file format of Images
    TrNum = 8; %Training Number  
    TsNum=2; % Testing Number
    sample_set=TsNum+TrNum; % sample set size
    ind = 1:total_sample;

%% Creating Database of Training and Testing Images ORL

    Excluded_Image =ind(TrNum+1:total_sample); 
    Tr = Create_Database_ORL(num_subjects,DatabasePathORL,ff,Excluded_Image);
    Excluded_Image =[ind(1:TrNum) ind(sample_set+1:total_sample)]; 
    Ts = Create_Database_ORL(num_subjects,DatabasePathORL,ff,Excluded_Image); 
    
    
%% Creating Database of Training and Testing Images UMIST

%     Excluded_Image=ind(TrNum+1:total_sample); 
%     Tr = Create_Database_UMIST(num_subjects,DatabasePathUMIST,ff,Excluded_Image);
%     Excluded_Image=[ind(1:TrNum) ind(sample_set+1:total_sample)]; 
%     Ts = Create_Database_UMIST(num_subjects,DatabasePathUMIST,ff,Excluded_Image);
    
    
%% Performing PCA_Test
    
    TrainFaces = PCA_2D(Tr,dimensions);
    TestFaces = PCA_2D(Ts,dimensions);
    EigenFaces = [TrainFaces ; TestFaces];
     
    
 %% Reshaping Features obtained from 2D- PCA
    
    input = [];
    for i = 1:size(EigenFaces,1)
        input = [input reshape(EigenFaces(i,:,:),size(EigenFaces,2)*size(EigenFaces,3),1)];
    end
    
%% Target Output Calculation
 
  target = Target_Calculation(num_subjects,TrNum,TsNum);

%% Random Sequence Generator

  max_accu_fi = -inf;
  count_correct = 0;
  temp_accu = [];
  accuracy = [];
  
%% Rand Sequence
  for seq = 1:5
     
    rng('shuffle');  
    rand_seq = randperm(sample_set*num_subjects);  
    [train_input,train_target,test_input,test_target] = Rand_Seq_Input(input, target, sample_set, rand_seq, num_subjects, TrNum);
       
%% Neural Network Classifier
    
    [net,b,iw,lw,optimal_hidden_neuron,max_accu] = classifier(train_input, train_target, test_input, test_target);
      
%% Custom Testing of Neural Network with Floating Point Representation

    float_accu = Floating_Point_Testing(b, iw, lw, test_input, test_target, optimal_hidden_neuron, num_subjects, TsNum);
    
%% Custom Testing of Neural Network with Fixed Point Representation
         
     [opti_parameter,fixed_accu,fixed_approx_accu,temp_accu]= Fixed_Point_Testing(b, iw, lw, test_input, test_target, optimal_hidden_neuron, num_subjects, TsNum, max_accu_fi, float_accu, max_accu);      
     
  end
  
%% Plotting of various accuracy

%    figure;
%    temp_accu = [temp_accu [100*(1-fixed_approx_accu), 100*(1-float_accu), max_accu]];
%    accuracy(init_param,:) = temp_accu;
%    barh(1:1:num_init_param,accuracy,'grouped');
%    title('Random Sequence');
%    xlabel('Accuracy in percentage');
%    ylabel('Random Initial Weights');
%    legend('8:6','8:6 Approx','Float','Inbuilt','Location','southwest');
%   end
