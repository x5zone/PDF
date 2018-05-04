function NN_MLP_Lorenz
% Exp 9: MLP for dynamic reconstruction of Lorenz attractor

% Copyright
% Yanbo Xue
% Adaptive Systems Laboratory
% McMaster University
% yxue@soma.mcmaster.ca
% May 2, 2007


clear;
% %%======================= Generating the data =============================
% sig = 10;
% b = 8/3;
% r = 28;
% g = inline('[ 10*(xx(2)-xx(1)) ; 28*xx(1)-xx(2)-xx(1)*xx(3) ; xx(1)*xx(2)-(8/3)*xx(3) ]', 't','xx');
% %[t,xx] = ode23(g,[0,400],[1;2;3]);
% [t,xx] = ode15s(g,[0,400],[1;2;3]);
% %plot3(x(:,1),x(:,2),x(:,3));
% %plot(t(100:end),x(100:end,1))
% plot(xx(:,1))
% %t = 1:0.01:20;
% %x = sin(t);
% %sig_lorenz = x';
% %sig_lorenz = xx(100:end,1);
load sig_lorenz_full.mat;

%sig_lorenz = sig_lorenz(200:end);

%%========= Preprocess the input data : remove mean and normalize =========
mean1 = mean(sig_lorenz);
for i = 1:length(sig_lorenz),
    nor_data(i) = sig_lorenz(i) - mean1;
end
%nor_data = sig_lorenz;
max1  = max(abs(nor_data));
for i = 1:length(sig_lorenz),
    sig_lorenz(i) = nor_data(i)./max1;
end
sig_lorenz = sig_lorenz;

%%====================== Parameters setup =================================
%load data_shuffled1.mat; % shuffled data for half-moon with distance d = 1
n_in = 20;      %number of input neurons
n_hd = 200;      %number of hidden neurons
n_ou = 1;       %number of output neurons
len_subset = n_in + n_ou;
S_point  = 0;   %Starting point of the testing data generation
                %The first testing data should be at S_point+len_subset-1
t = [S_point : S_point+len_subset-1];% Vector t for generating testing data

%% Step 1: Initialization of Multilayer Perceptron (MLP)
w1{1} = 0.1*(rand(n_hd,n_in+1)./2  - 0.25); % initial weights of dim: n_hd x n_in between input layer to hidden layer
dw0{1} = zeros(n_hd,n_in+1); % rand(n_hd,n_in)./2  - 0.25;
%w1{1} = rand(n_hd,n_in)./2  - 0.25; % initial weights of dim: n_hd x n_in between input layer to hidden layer
%dw0{1} = zeros(n_hd,n_in); % rand(n_hd,n_in)./2  - 0.25;
w1{2} = 0.1*(rand(n_ou,n_hd)./2  - 0.25); % initial weights of dim: n_ou x n_hd between hidden layer to output layer
dw0{2} = zeros(n_ou,n_hd); %rand(n_ou,n_hd)./2  - 0.25;
n_tr  = 700; % number of training sets
n_te  = 800;  % number of testing sets
num_Epoch  = 50;    % number of epochs
mse_thres  = 1E-4;    % MSE threshold
mse_train  = Inf;     % MSE for training data
epoch = 1;
alpha = 0;         % momentum constant
%eta2  = 0.1;         % learning-rate for output weights
%eta1  = 0.1;          % learning-rate for hidden weights

eta1 = annealing(1E-1,1E-5,num_Epoch);
eta2 = annealing(1E-1,1E-5,num_Epoch);

cc=cputime;
%% Main Loop for Training
while mse_train > mse_thres && epoch <= num_Epoch
    %% using all data for training for this epoch
    cc1=cputime;
    %fprintf('Epoch #: %d ->',epoch);
    for i = 1:n_tr,
        %% Forward computation
        
        %y = sig_lorenz(t+epoch);
        y = sig_lorenz(t+i);
        x = [y(1:len_subset-1);1];
        %x = y(1:len_subset-1);
        d(i) = y(len_subset); % desired output
        hd  = hyperb(w1{1}*x);    % hidden neurons are nonlinear
        o(i)= hyperb(w1{2}*hd);% output neuron is nonlinear
        %o(i) = w1{2}*hd;
        e(i)= d(i) - o(i);
        
        %% Backward computation
        delta_ou = e(i).*d_hyperb(w1{2}*hd);              % delta for output layer
        delta_hd = (w1{2}'*delta_ou).*d_hyperb(w1{1}*x);  % delta for hidden layer
        dw1{1} = delta_hd*x';
        dw1{2} = delta_ou*hd';
        
        %% weights update
        w2{1} = w1{1} + alpha*dw0{1} + eta1(epoch)*dw1{1}; % weights input -> hidden
        w2{2} = w1{2} + alpha*dw0{2} + eta2(epoch)*dw1{2};  % weights hidden-> output
        
        %% move weights one-step
        dw0 = dw1;
        w1 = w2;
        
        mse_temp = mean(e.^2);
        %t = t + len_subset;
    end
    mse(epoch) = mean(e.^2);
    mse_train = mse(epoch);
    %fprintf('MSE = %f\n',mse_train);
    %fprintf('It took %4.1f seconds\n',cputime-cc1);
%     if epoch >=2,
%        if mse(epoch) > mse(epoch-1),
%            break;
%        end
%     end
    epoch = epoch + 1;
end
fprintf('Training is done. It took %4.1f seconds\n',cputime-cc);

%for i = n_tr+1:n_tr+1+n_te,
%% Show trainig results
for i = 1 : n_tr,
    y = sig_lorenz(t+i);
    x = [y(1:len_subset-1);1];
    %x = y(1:len_subset-1);
    d(i) = y(len_subset); % desired output
    hd = hyperb(w1{1}*x);
    o(i) = hyperb(w1{2}*hd);
    %o(i) = w1{2}*hd;
    %t = t + 1;
end

%% Show testing results
for i = n_tr+1 : n_tr+1+n_te,
    y = sig_lorenz(t+i);
    x = [y(1:len_subset-1);1];
    %x = y(1:len_subset-1);
    d_t(i) = y(len_subset); % desired output
    hd = hyperb(w1{1}*x);
    o_t(i) = hyperb(w1{2}*hd);
    %o(i) = w1{2}*hd;
    %t = t + 1;
end

%% Show recursive prediction results
x_ini = sig_lorenz(1:n_in);
%dd(1) = y(len_subset);
%oo(1) = dd(1);
for i = 1 : n_tr,
    y = sig_lorenz(t+i);
    x = [x_ini;1];
    %x = x_ini;
    hd = hyperb(w1{1}*x);
    oo(i) = hyperb(w1{2}*hd);
    %oo(i) = w1{2}*hd;
    dd(i) = y(len_subset);
    x_ini = [x_ini(2:n_in,:);oo(i)];
end

% figure;
% plot(d);
% hold on;
% plot(o,'r--');
% title('Training');
% xlabel('Time');ylabel('Amplitude');

figure;
plot((max1*d_t(n_tr+1 : n_tr+1+n_te))+mean1,'k');
hold on;
plot((max1*o_t(n_tr+1 : n_tr+1+n_te))+mean1,'r--');
title('Testing');
legend('Theory','Prediction');
xlabel('Time');ylabel('Amplitude');

% figure;
% plot(dd);
% hold on;
% plot(oo,'r--');
% title('Recursive Prediction');
% xlabel('Time');ylabel('Amplitude');

figure;
plot(mse,'k');
xlabel('Number of epochs');ylabel('MSE');
title('MSE');