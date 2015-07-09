function Eigenfaces = PCA_2D(Tr,dimensions)


%% Mean of Training Data
M=mean(Tr,1);

%% Removing Mean from the Faces

 A=Tr-repmat(M,size(Tr,1),1);
 

 %% Find Covariance Matrix's Eigenvector 
 L =zeros(size(A,3),size(A,3));
 for i = 1:size(A,1) 
     temp = reshape(A(i,:,:),size(A,2),size(A,3));
     L = L + temp'*temp; 
 end
 L = L/size(A,1);

 [U,D]=eig(L);% Eigen Vectors of Surogate Matrix

 %% Sorting Eigenvectors 
 Significant_Eig_Vec = [];
 eigValue=diag(D);
 [~,ind]=sort(eigValue,'descend');
 Significant_Eig_Vec = U(:,ind);

 Significant_Eig_Vec = Significant_Eig_Vec(:,1:dimensions);
 
 for i =1:size(A,1)
     sample_image = reshape(Tr(i,:,:),size(Tr,2),size(Tr,3));
     Eigenfaces(i,:,:)= sample_image*Significant_Eig_Vec;
 end


end