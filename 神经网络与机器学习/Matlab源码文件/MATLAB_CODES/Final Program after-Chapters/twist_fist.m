function data_shuffled = twist_fist(d1,d2,d3,n_samp)
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

% d1 = 10;
% d2 = 5;
% d3 = 2;
% n_samp = 6000;

width1 = d1 -d2;
width2 = d2 - d3;

%%=== Generating data for two classes ==
aa = rand(2,n_samp);
radius = d1*aa(1,:);
theta = 2*pi*aa(2,:);
x = radius.*cos(theta);
y = radius.*sin(theta);
label = 0*ones(1,length(x));

for i = 1: length(x),
    if (((radius(i) >= d2)&&( radius(i) <= d1)) && ((theta(i) >= 0)&&(theta(i) <= pi)))...
            || (((radius(i) >= d3)&&( radius(i) <= d2)) && ((theta(i) >= pi)&&(theta(i) <= 2*pi))) ...
            || (((radius(i)>=0)&&(radius(i)<=d3)) && ((theta(i)>=0)&&(theta(i)<=pi)) )
        label(i) = 1;
    else
        label(i) = -1;
    end
end

% figure;
% subplot(221);
% box on;
% title(['training samples: d1 = ', num2str(d1), ' d2 = ', num2str(d2), ' d3 = ', num2str(d3)]);
% hold on;
% for i = 1: length(x),
%     if label(i) == 1,
%         plot(x(i),y(i),'rx');
%     else
%         plot(x(i),y(i),'k+');
%     end
% end
% xlim([-d1,d1]); ylim([-d1,d1]);
% axis('equal');



% if rad < width/2,
%     error('The radius should be at least larger than half the width');
% end
% 
% if mod(n_samp,2)~=0,
%     error('Please make sure the number of samples is even');
% end

% aa = rand(2,n_samp/2);
% radius = (rad-width/2) + width*aa(1,:);
% theta = pi*aa(2,:);


% x     = radius.*cos(theta);
% y     = radius.*sin(theta);
% label = 1*ones(1,length(x));  % label for Class 1
% 
% x1    = radius.*cos(-theta) + rad;
% y1    = radius.*sin(-theta) - d;
% label1= -1*ones(1,length(x)); % label for Class 2

data_shuffled = [x;
                 y;
                 label];

% data  = [x, x1;
%          y, y1;
%          label, label1];
%      
% [n_row, n_col] = size(data);
% 
% shuffle_seq = randperm(n_col);
% 
% for i = (1:n_col),
%     data_shuffled(:,i) = data(:,shuffle_seq(i));
% end;