function y = hyperb(x)
% y = hyperb (x)
% hyperbolic function
% x - input data
% y - output data

%%%% Author: Yanbo Xue & Le Yang
%%%% ECE, McMaster University
%%%% yxue@grads.mcmaster.ca; yangl7@psychology.mcmaster.ca
%%%% May 11, 2006
%%%% This is a joint work by Yanbo and Le
%%%% For Project of Course of Dr. Haykin: Neural Network

y = (exp(2*x)-1)./(exp(2*x)+1);

%y = 1./(1+exp(-x));