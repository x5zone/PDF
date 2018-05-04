function classifier=rlsc(options,data)
%
% RLSC Regularized Least Squares Classifier
%
% classifier = rlsc(options,data,labels)
% 
% Inputs: 
%
% options -- a data structure with the following fields
% (Use ml_options to make this structure. Type help ml_options)
%  
% options.Kernel = 'linear' | 'poly' | 'rbf'
% options.KernelParam = 0 | degree | gamma | dosent_matter
% options.gamma_A == regularization parameters 
% 	Note: To do semi-supervised inference 
%             use options.PointCloud, options.DeformationMatrix
%             and pass the semi-supervised kernel matrix in data.K
% operationally, this code only uses options.gamma_A though options
% should have correct fields since these are saved in the 
% output "classifier" data structure
%  
% data -- Input data structure with the following fields
%
%     data.X = a N x D data matrix
%     (N examples, each of which is a D-dimensional vector)
%  
%     data.Y = a N x 1 label vector (-1,+1)
%
%     data.K = a N x N kernel gram matrix (optional)
%
% 
% Output:
%
% classifier : A data structure that encodes the binary classifier. 
% (Type 'help saveclassifier'). 
%
% Notes: Since all kernel classifiers are  of the form 
% 	      sum_i alpha_i K(x,x_i) + b, 
%         we save training examples x_i, alpha_i and b in "classifier".
% 
% (a) To predict on new examples, use 
%                predict(classifier,test_data,test_data_labels)
% (b) For training on multiclass problems, use 
%                multiclass(...)
%
% Author:  
%    Vikas Sindhwani (vikass@cs.uchicago.edu)
%
%
%
 
disp('Training Regularized Least Squares Classifier');

if ~isfield(data,'K')
   error('Kernel Gram matrix not found');
end  
% this is all RLS does ... invert a matrix
      I=eye(size(data.K,1));
      alpha=(data.K+options.gamma_A*I)\data.Y;
% set a bias to threshold
bias=1;
if isfield(options,'bias'), bias=options.bias; end
 
   if bias==0, b=0; end
   if (bias > 0) & (bias < 1), 
	f=data.K*alpha;g=sort(f); 
        jj=ceil((1-bias)*length(f)); 
        b=0.5*(g(jj)+g(jj+1)); 
   end
   if bias==1, f=data.K*alpha; b=-mean(data.Y-f); end
% save classifier
classifier=saveclassifier('rlsc',1:length(data.Y),alpha,data.X,b,options);
