function out = mysign(d)
% a signsum function 

% Copyright
% Yanbo Xue
% Adaptive Systems Laboratory
% McMaster University
% yxue@soma.mcmaster.ca
% Feb. 22, 2007

out = 1*(d>=0) + (-1)*(d<0);

% if distance(d,1) <= distance(d,0),
%     out = 1;
% else
%     out = 0;
% end

return