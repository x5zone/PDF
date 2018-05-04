function [data, data_shuffled] = halfmoon(rad,width,d,n_samp)
% A function to generate the halfmoon data
% where Input:
%         rad  - central radius of the half moon
%        width - width of the half moon
%           d  - distance between two half moon
%      n_samp  - total number of the samples
%       Output:
%         data - output data
%data_shuffled - shuffled data
% For example
% halfmoon(10,2,0,1000) will generate 1000 data of 
% two half moons with radius [9-11] and space 0.

% Copyright
% Yanbo Xue
% Adaptive Systems Laboratory
% McMaster University
% yxue@soma.mcmaster.ca
% Feb. 22, 2007

if rad < width/2,
    error('The radius should be at least larger than half the width');
end

if mod(n_samp,2)~=0,
    error('Please make sure the number of samples is even');
end

aa = rand(2,n_samp/2);
radius = (rad-width/2) + width*aa(1,:);
theta = pi*aa(2,:);


x     = radius.*cos(theta);
y     = radius.*sin(theta);
label = 1*ones(1,length(x));  % label for Class 1

x1    = radius.*cos(-theta) + rad;
y1    = radius.*sin(-theta) - d;
label1= -1*ones(1,length(x)); % label for Class 2

data  = [x, x1;
         y, y1;
         label, label1];
     
[n_row, n_col] = size(data);

shuffle_seq = randperm(n_col);

for i = (1:n_col),
    data_shuffled(:,i) = data(:,shuffle_seq(i));
end;