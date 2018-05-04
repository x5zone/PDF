function [Ktilde,K_test_tilde]=Deform(r,K,M,K_test)
% Computes the semi-supervised Kernel
% [Ktilde,K_test_tilde]=Deform(r,K,M,K_test)
% Inputs:
% K: the gram matrix of a kernel over labeled+unlabeled data (nxn matrix)
% M: a graph regularizer (nxn  matrix)
% K_test: the gram matrix of a kernel between training and test points
% (optional) size m x n for m test points.
% r: deformation ratio (gamma_I/gamma_A)
% Outputs:
% Ktilde: the gram matrix of the semi-supervised deformed kernel over labeled+unlabeled
% data
% K_test_tilde: the gram matrix of the semi-supervised deformed kernel
% between training and test points

I=eye(size(K,1));
Ktilde=(I+r*K*M)\K;
if exist('K_test','var')
    K_test_tilde=(K_test - r*K_test*M*Ktilde);
end