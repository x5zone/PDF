function NN_LS_regularization(dist,mu) 
% Exp 1: Classificaiton using Regularized Least Square

% Copyright
% Yanbo Xue
% Adaptive Systems Laboratory
% McMaster University
% yxue@soma.mcmaster.ca
% Feb. 22, 2007

%%================== Step 0: Generating halfmoon data =====================
%clear;
%load data_shuffled1.mat; % shuffled data for half-moon with distance d = 1
rad    = 10;   % central radius of the half moon
width  = 6;    % width of the half moon
%dist   = -1;   % distance between two half moons
num_tr = 5000; % number of training sets
num_te = 2000;  % number of testing sets
num_samp = num_tr+num_te;% number of samples
epochs = 1;
fprintf('Regularized Least Square for Classification\n');
fprintf('_________________________________________\n');
fprintf('Generating halfmoon data ...\n');
fprintf('  ------------------------------------\n');
fprintf('  Points generated: %d\n',num_samp);
fprintf('  Halfmoon radius : %2.1f\n',rad);
fprintf('  Halfmoon width  : %2.1f\n',width);
fprintf('      Distance    : %2.1f\n',dist);
fprintf('  Number of epochs: %d\n',epochs);
fprintf('  ------------------------------------\n');
seed=2e5;
rand('seed',seed);
[data, data_shuffled] = halfmoon(rad,width,dist,num_samp);

%%============= Step 1: Initialization of Perceptron network ==============
num_in = 2;    % number of input neuron
b      = 0;  % bias
err    = 0;    % a counter to denote the number of error outputs
%eta  = 0.95; % learning rate parameter
eta = annealing(0.9,1E-5,num_tr);
%w    = [b;zeros(num_in,1)];% initial weights
%mu = 0.1;
I = eye(num_in);

%%=========================== Main Loop ===================================
%% Step 2,3: activation and actual response
st = cputime;
fprintf('Calculating weights using LS ...\n');
fprintf('  ------------------------------------\n');

%%==== Step 1: Preprocess the input data, remove mean and normalize =======
mean0 = [mean(data(1:2,:)')';0];         % mean of the original data
for i = 1:num_samp,
    data_norm(:,i) = data_shuffled(:,i) - mean0;
end
max0  = [max(abs(data_norm(1:2,:)'))';1];% max of the original data
for i = 1:num_samp,
    data_norm(:,i) = data_norm(:,i)./max0;
end

data_shuffled = data_norm;


A = data_shuffled(1:2,:)';
yy= data_shuffled(3,:)';

w = inv(A'*A + mu*I)*A'*yy;
%w = pinv(R)*r;
%w = R\r;

 w = [b;w];
% fprintf('  Points trained : %d\n',num_tr);
% fprintf('       Time cost : %4.2f seconds\n',cputime - st);
% fprintf('  ------------------------------------\n');
% 
% %%=============== Plotting Learning Curve =================================
% figure;
% plot(mse,'k');
% title('Learning curve');
% xlabel('Number of epochs');ylabel('MSE');

% figure;
% hold on;
% for i = 1 : num_tr,
% %for i = num_tr+1:num_samp,
%     x = [1 ; data_shuffled(1:2,i)]; % fetching data for testing
%     %x = [data_shuffled(1:2,i+num_tr)];
%     y(i) = mysign(w'*x);
%     %y(i) = w'*x;
%     if y(i) == 1 ,
%         plot(x(2),x(3),'rx');
%     end
%     if y(i) == -1,
%         plot(x(2),x(3),'k+');
%     end
% end
% title(['\mu = ', num2str(mu)]);

% %%==== Step 1: Preprocess the input data, remove mean and normalize =======
% for i = 1:num_samp,
%     data_norm(:,i) = data_shuffled(:,i).*max0;
% end
% for i = 1:num_samp,
%     data_norm(:,i) = data_norm(:,i) + mean0;
% end
% 
% data_shuffled = data_norm;

%%================= Colormaping the figure here ===========================
%%=== In order to avoid the display problem of eps file in LaTeX. =========
figure;
hold on;
xmin = min(data_shuffled(1,:));
xmax = max(data_shuffled(1,:));
ymin = min(data_shuffled(2,:));
ymax = max(data_shuffled(2,:));
[x_b,y_b]= meshgrid(xmin:(xmax-xmin)/100:xmax,ymin:(ymax-ymin)/100:ymax);
z_b  = 0*ones(size(x_b));
%wh = waitbar(0,'Plotting testing result...');
for x1 = 1 : size(x_b,1)
    for y1 = 1 : size(x_b,2)
        input = [1; x_b(x1,y1); y_b(x1,y1)];
        %input = [x_b(x1,y1); y_b(x1,y1)];
        z_b(x1,y1) = w'*input;
    end
    %waitbar((x1)/size(x,1),wh)
    %set(wh,'name',['Progress = ' sprintf('%2.1f',(x1)/size(x,1)*100) '%']);
end
%% Adding colormap to the final figure
%figure;
% Scale data points back to original value
x_b = x_b*max0(1) + mean0(1);
y_b = y_b*max0(2) + mean0(2);
xmin = xmin*max0(1) + mean0(1); % scale xmin
xmax = xmax*max0(1) + mean0(1); % scale xmax
ymin = ymin*max0(2) + mean0(2); % scale ymin
ymax = ymax*max0(2) + mean0(2); % scale ymax
sp = pcolor(x_b,y_b,z_b);
load red_black_colmap;
%colormap(myhot_light);
colormap(red_black);
shading flat;

%%============================== Testing ==================================
fprintf('Testing the classifier using LS ...\n');
for i = 1 : num_te,
%for i = num_tr+1:num_samp,
    x = [1 ; data_shuffled(1:2,i+num_tr)]; % fetching data for testing
    %x = [data_shuffled(1:2,i+num_tr)];
    y(i) = mysign(w'*x);
    %y(i) = w'*x;
    x(2) = x(2)*max0(1) + mean0(1);
    x(3) = x(3)*max0(2) + mean0(2);
    if y(i) == 1 ,
        plot(x(2),x(3),'rx');
    end
    if y(i) == -1,
        plot(x(2),x(3),'k+');
    end
end
xlabel('x');ylabel('y');
title(['Classification using LS with dist = ',num2str(dist), ...
       ', radius = ', num2str(rad), ', \lambda = ', num2str(mu), ' and width = ',num2str(width)]);
fprintf('Mission accomplished!\n');
% Calculate testing error rate
for i = 1:num_te,
    if abs(y(i) - data_shuffled(3,i+num_tr)) > 1E-6,
        err = err + 1;
    end
end
fprintf('  ------------------------------------\n');
fprintf('   Points tested : %d\n',num_te);
fprintf('    Error points : %d (%5.2f%%)\n',err,(err/num_te)*100);
fprintf('  ------------------------------------\n');
fprintf('_________________________________________\n');

%%======================= Plot decision boundary ==========================
set(gca,'XLim',[xmin xmax],'YLim',[ymin ymax]);
%% Adding contour to show the boundary
contour(x_b,y_b,z_b,[0 0],'k','Linewidth',1);
%contour(x_b,y_b,z_b,[-1 -1],'k:','Linewidth',2);
%contour(x_b,y_b,z_b,[1 1],'k:','Linewidth',2);
set(gca,'XLim',[xmin xmax],'YLim',[ymin ymax]);
grid on;
%% That's all, folks
