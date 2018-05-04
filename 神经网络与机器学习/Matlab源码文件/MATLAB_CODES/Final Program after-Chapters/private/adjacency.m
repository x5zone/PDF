function A = adjacency(options,X,step);

% ADJACENCY  Computes the Graph Laplacian of the adjacency graph
%            of a data set X
%
% A=adjacency(options,X)
%
% Inputs:
%
% X is a N x D data matrix
%  (N examples, each of which is a D-dimensional vector)
%
% options: a data structure with the following fields
%                       (type  help ml_options)
%
% options.NN = integer (number of nearest neighbors to use)
% options.GraphDistanceFunction: 'euclidean' | 'cosine'
%       (distance function used to make the graph)
%
%
% step : number of examples to process per block computation
%
% Output
%  A : sparse symmetric NxN NN-adjacency matrix 
% 
%
% Author:
% Vikas Sindhwani (vikass@cs.uchicago.edu)



fprintf(1, ...
['Computing ' num2str(options.NN) '-NN  '  options.GraphDistanceFunction  ' Adjacency Graph']);
  

n = size(X,1);

p=2:(options.NN+1);
if size(X,1)<500
    step=size(X,1);
else
	step=500;
end
 frac=ceil(n/step);

 qq=[];ZZ=[]; t=0; T=0;
R=repmat((1:step)',[options.NN 1]);
  for i1=1:step:n    
 t=t+1;
    i2 = i1+step-1;
    if (i2> n) 
      i2=n;
      R=repmat((1:(i2-i1+1))',[options.NN 1]);
    end;

    XX= X(i1:i2,:);  
    dt = feval(options.GraphDistanceFunction, XX',X');
    [Z,I] = sort ( dt,2);
	 	    
  i=i2-i1+1;
  Z=Z(:,p); Z=Z';% i1:i2
  I=I(:,p);I=I';
  qq=[qq;I(:)];
  ZZ=[ZZ; Z(:)];

fprintf(1,'...%d%%', ceil((t/frac)*100));


 end 
fprintf(1,'\n');
I=repmat((1:n),[options.NN 1]); I=I(:);

 if strcmp(options.GraphDistanceFunction,'cosine')
	A=sparse(I,qq,ZZ.*(ZZ<1),n,n);
 else
	A=sparse(I,qq,ZZ,n,n);

 end

% symmetrize
A=A+((A~=A').*A');
