function out = annealing(start_data,end_data,num)
% annealing - anneal data from 'start' to 'end' with number 'num'
% out = annealing(start,end,num)
% start_data - starting point
% end_data   - ending point
% num        - number of annealing point
% out        - annealed data sequence

%%%% Author: Yanbo Xue & Le Yang
%%%% ECE, McMaster University
%%%% yxue@grads.mcmaster.ca; yangl7@psychology.mcmaster.ca
%%%% May 12, 2006
%%%% This is a joint work by Yanbo and Le
%%%% For Project of Course of Dr. Haykin: Neural Network

% Check input parameters
if start_data == end_data, error ('Starting point and ending point is the same.'); end;
if num <= 2, error('Number of annealed data point should > = 2.'); end;

% Linear annealing
step = (end_data - start_data)/(num-1);
out = [start_data:step:end_data];