function [L,options] = laplacian(options,X)  

% LAPLACIAN  Computes the Graph Laplacian of the adjacency graph
%            of a data set X
%      
% [L,options]=laplacian(options,data)
% 
% Inputs: 
% 
% X is a N x D data matrix 
%  (N examples, each of which is a D-dimensional vector)  
%
% options: a data structure with the following fields 
%			(type  help ml_options)
%
% options.NN = integer (number of nearest neighbors to use)
% options.DISTANCEFUNCTION: 'euclidean' | 'cosine'  
%   	(distance function used to make the graph)
% options. WEIGHTTYPPE='binary' | 'heat'
%       (binary weights or heat kernel weights)         
% options.WEIGHTPARAM= width for heat kernel | 'default'
%                      (default uses mean edge length distance)
% options.NORMALIZE= 0 | 1 
%          (0 for un-normalized and 1 for normalized graph laplacian) 
%
% Output
%  L : sparse symmetric NxN matrix 
%  options: updated options structure (options)
%
% Notes: Calls adjacency.m to construct the adjacency graph. This is 
%        fully vectorized for fast performance.
%
% Author:
% Vikas Sindhwani (vikass@cs.uchicago.edu)
% 		[modified from Misha Belkin's code]
% 


tic;

 fprintf(1, 'Computing Graph Laplacian\n');
 	W = adjacency(options,X);
 fprintf(1,'\n');


% disassemble the sparse matrix
	[A_i, A_j, A_v] = find(W);

switch options.GraphWeights
      
	case 'binary'
		 fprintf(1,'Using Binary weights\n');
 		  W=sparse(A_i,A_j,1,size(W,1),size(W,1));  
 
	case 'heat'     
    		 if isstr(options.GraphWeightParam) % default
       			 ind=find(W);
    	   	         t=full(mean(W(ind)));
                	 options.GraphWeightParam=t;
        	 else
       			 t=options.GraphWeightParam;
        	 end
    	       fprintf(1,['Using heat kernel of width ' num2str(t) '\n']);
    
                 W=sparse(A_i,A_j,exp(-A_v.^2/(2*t*t)));
    
	otherwise
    		 error('Unknown Weighttype\n');   
	end

   D = sum(W(:,:),2);  
   
   if options.GraphNormalize==0
        L = spdiags(D,0,speye(size(W,1)))-W; 
   else % normalized laplacian
        fprintf(1, 'Normalizing the Graph Laplacian\n');
        D(find(D))=sqrt(1./D(find(D)));
        D=spdiags(D,0,speye(size(W,1)));
        W=D*W*D;
        L=speye(size(W,1))-W;
   end

fprintf(1,['Graph Laplacian computation took ' num2str(toc) '  seconds.\n']);
