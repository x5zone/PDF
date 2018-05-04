function alpha=laprls(K,Y,M,gamma_A,gamma_I)
% RLS  Laplacian Regularized Least Squares
% INPUTS:
% K: n X n gram matrix
% M: n x n Laplacian matrix
% Y: n x 1 labels vector -- unlabeled entries are 0
% lambda: regularization parameter
% OUTPUTS:
% alpha: n X 1 expansion cofficients
%
% Vikas Sindhwani (vikass@cs.uchicago.edu)


n=size(K,1);
l=length(Y);

I=eye(n);
J=diag(abs(Y));

alpha=(J*K+gamma_A*I + gamma_I*M*K)\Y;