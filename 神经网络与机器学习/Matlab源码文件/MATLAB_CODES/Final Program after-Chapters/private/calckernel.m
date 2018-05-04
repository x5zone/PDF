function K=calckernel(options,X1,X2);

% CALCKERNEL Computes Gram matrix of a specified kernel on a given 
% set of points X1 or between X1 and X2. This code is fully vectorized. 
% 
% K=grammatrix(options,X1); 
% K=grammatrix(options,X1,X2); 
% 
% 
% Inputs: 
% 
% X1 (X2 is optional) = a N x D data matrix 
% (N examples, each of which is a D-dimensional vector) 
% 
% options = a data structure with the following fields 
% (Use ml_options to make this  structure. Type help ml_options) 
% 
% options.Kernel = 'linear' | 'poly' | 'rbf' 
% 
% options.KernelParam = specifies parameters for the kernel functions: 
% 			degree for 'poly'; sigma for 'rbf'; 
% 			can be ignored for linear kernel 
% 
%  option.PointCloud, options.DeformationMatrix 
% 		These fields implement  the semisupervised data-dependent 
%               kernel  proposed in the paper: 
% 		"Beyond the point Cloud: from Transductive to  Semi-supervised 
%	         Learning", V. Sindhwani, P. Niyogi, M. Belkin, 2005 
% 	         where the kernel specified through options.Kernel and 
%               options.KernelParam is deformed using a point-cloud norm. 
% 
% options.PointCloud : n x D matrix 
% 		(n data points of dimensionality D that make the point cloud) 
% option.DeformationMatrix : The (n x n) matrix M in the paper that defines 
%			      the point cloud norm. 
% 
%  Note: even X1 is optional if options.PointCloud=X1
%        then the gram matrix of the modified kernel can be computed faster
%  
% Outputs: 
% 
%  Given a single data matrix X1 (N xD) 
%  returns Gram matrix K (N x N) 
% 
%  Given two data matrices X1 (N1 x D), X2 (N2 x D) 
%  returns Gram matrix K (N2 x N1) 
%
%  Author: 
%  Vikas Sindhwani (vikass@cs.uchicago.edu) 
%

  tic;


kernel_type=options.Kernel;
kernel_param=options.KernelParam;

if exist('X1','var')
 dim=size(X1,2);
 n1=size(X1,1);
end

if exist('X2','var') 
    n2=size(X2,1);
end


if ~isfield(options,'PointCloud') 
%% so we dont intend to deform the kernel
 
fprintf(1, 'Computing %s Kernel', kernel_type);

switch kernel_type

case 'linear'
    fprintf(1,'\n');
    if exist('X2','var') 
        K=X2*X1';  
    else
        K=X1*X1';
    end
    
case 'poly'
    fprintf(1, ' of degree %s\n', kernel_param);
    if exist('X2','var') 
        K=(X2*X1').^kernel_param;
    else
        K=(X1*X1').^kernel_param;
    end
    
case 'rbf'  
    fprintf(1, ' of width %f\n', kernel_param);
    if exist('X2','var') 
        K = exp(-(repmat(sum(X1.*X1,2)',n2,1) + repmat(sum(X2.*X2,2),1,n1) ...
            - 2*X2*X1')/(2*kernel_param^2)); 
    else
    
        P=sum(X1.*X1,2);
        K = exp(-(repmat(P',n1,1) + repmat(P,1,n1) ...
            - 2*X1*X1')/(2*kernel_param^2)); 
        
    end

otherwise
 
		error('Unknown Kernel Function.');
end

    
else % we intend to deform our kernel
     % this code can be speeded up later
    
    opt.Kernel=options.Kernel;
    opt.KernelParam=options.KernelParam;
    X=options.PointCloud;
    G=calckernel(opt,X);
    if isfield(options,'DeformationMatrix')
    	M=options.DeformationMatrix;
    else % use the laplacian
        M=laplacian(options,X);
disp(['Using Iterated Laplacian of Degree ' num2str(options.LaplacianDegree)]);
M=(options.gamma_I/options.gamma_A)*(mpower(M,options.LaplacianDegree));
     
end

    I=eye(size(G,1)); 
    if exist('X1','var')
                A=calckernel(opt,X1,X); % n x n1
        	if exist('X2','var')
      	           B=calckernel(opt,X2,X); %n x n2
       	           K1=calckernel(opt,X1,X2); % n2 x n1
                else
                   B=A;
                   K1=calckernel(opt,X1);
                end
      disp('Deforming the Kernel');	     
     
      
      K=(K1 - B'*((I+M*G)\M)*A);
               
    else % we need gram matrix for the modified kernel over the point cloud
      disp('Deforming the Kernel');     
      K=(I+G*M)\G;
   
    end 


end    

disp(['Computation took ' num2str(toc) ' seconds.']);
