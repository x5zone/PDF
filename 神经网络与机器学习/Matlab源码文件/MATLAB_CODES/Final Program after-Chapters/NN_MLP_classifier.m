function NN_MLP_classifier(dist)
% Exp 3(a): Multiple Layer Perceptron Classifier using Backpropagation

% Copyright
% Yanbo Xue
% Adaptive Systems Laboratory
% McMaster University
% yxue@soma.mcmaster.ca
% Feb. 22, 2007

%%=========== Step 0: Generating halfmoon data ============================
rad      = 10;   % central radius of the half moon
width    = 6;    % width of the half moon
%dist   = -4;   % distance between two half moons
num_tr   = 1000;   % number of training sets
num_te   = 2000;    % number of testing sets
num_samp = num_tr+num_te; % number of samples
fprintf('Multiple Layer Perceptron for Classification\n');
fprintf('_________________________________________\n');
fprintf('Generating halfmoon data ...\n');
fprintf('  ------------------------------------\n');
fprintf('  Points generated: %d\n',num_samp);
fprintf('  Halfmoon radius : %2.1f\n',rad);
fprintf('  Halfmoon width  : %2.1f\n',width);
fprintf('      Distance    : %2.1f\n',dist);
fprintf('  ------------------------------------\n');
[data, data_shuffled] = halfmoon(rad,width,dist,num_samp);

%%========== Step 1: Initialization of Multilayer Perceptron (MLP) ========
fprintf('Initializing the MLP ...\n');
n_in = 2;     % number of input neuron
n_hd = 20;    % number of hidden neurons
n_ou = 1;     % number of output neuron
%w = cell(2,1); 
%w1{1} = rand(n_hd,n_in+1)./2  - 0.25; % initial weights of dim: n_hd x n_in between input layer to hidden layer
w1{1} = rand(n_hd,n_in+1);
dw0{1}= zeros(n_hd,n_in+1); %rand(n_hd,n_in)./2  - 0.25;%
%w1{2} = rand(n_ou,n_hd+1)./2  - 0.25; % initial weights of dim: n_ou x n_hd between hidden layer to output layer
w1{2} = rand(n_ou,n_hd+1);
dw0{2}= zeros(n_ou,n_hd+1); %rand(n_ou,n_hd)./2  - 0.25;%
num_Epoch = 50;      % number of epochs
mse_thres = 1E-3;    % MSE threshold
mse_train = Inf;     % MSE for training data
epoch = 1;
alpha = 0;         % momentum constant
err    = 0;    % a counter to denote the number of error outputs
%eta2  = 0.1;         % learning-rate for output weights
%eta1  = 0.1;          % learning-rate for hidden weights
eta1 = annealing(0.1,1E-5,num_Epoch);
eta2 = annealing(0.1,1E-5,num_Epoch);

%%========= Preprocess the input data : remove mean and normalize =========
mean1 = [mean(data(1:2,:)')';0];
for i = 1:num_samp,
    nor_data(:,i) = data_shuffled(:,i) - mean1;
end
max1  = [max(abs(nor_data(1:2,:)'))';1];
for i = 1:num_samp,
    nor_data(:,i) = nor_data(:,i)./max1;
end

%%======================= Main Loop for Training ==========================
st = cputime;
fprintf('Training the MLP using back-propagation ...\n');
fprintf('  ------------------------------------\n');
while mse_train > mse_thres && epoch <= num_Epoch
    fprintf('   Epoch #: %d ->',epoch);
    %% shuffle the training data for every epoch
    [n_row, n_col] = size(nor_data);
    shuffle_seq = randperm(num_tr);
    nor_data1 = nor_data(:,shuffle_seq);
   
    %% using all data for training for this epoch
    for i = 1:num_tr,
        %% Forward computation
        x  = [nor_data1(1:2,i);1];     % fetching input data from database
        %d  = myint2vec(nor_data1(3,i));% fetching desired response from database
        d  = nor_data1(3,i);% fetching desired response from database
        hd = [hyperb(w1{1}*x);1];          % hidden neurons are nonlinear
        o  = hyperb(w1{2}*hd);         % output neuron is nonlinear
        e(:,i)  = d - o;
        
        %% Backward computation
        delta_ou = e(:,i).*d_hyperb(w1{2}*hd);            % delta for output layer
        delta_hd = d_hyperb(w1{1}*x).*(w1{2}(:,1:n_hd)'*delta_ou);  % delta for hidden layer
        dw1{1} = eta1(epoch)*delta_hd*x';
        dw1{2} = eta2(epoch)*delta_ou*hd';
              
        %% weights update
        w2{1} = w1{1} + alpha*dw0{1} + dw1{1};  % weights input -> hidden
        w2{2} = w1{2} + alpha*dw0{2} + dw1{2};  % weights hidden-> output
        
        %% move weights one-step
        dw0 = dw1;
        w1  = w2;
    end
    mse(epoch) =sum(mean(e'.^2));
    mse_train = mse(epoch);
    fprintf('MSE = %f\n',mse_train);
    epoch = epoch + 1;
end
fprintf('   Points trained : %d\n',num_tr);
fprintf('  Epochs conducted: %d\n',epoch-1);
fprintf('        Time cost : %4.2f seconds\n',cputime - st);
fprintf('  ------------------------------------\n');

%%=============== Plotting Learning Curve =================================
figure;
plot(mse,'k');
title('Learning curve');
xlabel('Number of epochs');ylabel('MSE');

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
        input = [(x_b(x1,y1)-mean1(1))/max1(1);(y_b(x1,y1)-mean1(2))/max1(2);1];
        hd= [hyperb(w1{1}*input);1];
        z_b(x1,y1) = hyperb(w1{2}*hd);
    end
    %waitbar((x1)/size(x,1),wh)
    %set(wh,'name',['Progress = ' sprintf('%2.1f',(x1)/size(x,1)*100) '%']);
end

%% Adding colormap to the final figure
%figure;
sp = pcolor(x_b,y_b,z_b);
load red_black_colmap;
colormap(red_black);
shading flat;
set(gca,'XLim',[xmin xmax],'YLim',[ymin ymax]);

%%========================== Testing ======================================
fprintf('Testing the MLP ...\n');
for i = num_tr+1:num_samp,
    x   = [nor_data(1:2,i);1];
    hd  = [hyperb(w1{1}*x);1];
    o(:,i)= hyperb(w1{2}*hd);
    xx  = max1(1:2,:).*x(1:2,:) + mean1(1:2,:);
    if o(:,i)>0%myvec2int(o(:,i)) == 1,
        plot(xx(1),xx(2),'rx');
    end
    if o(:,i)<0%myvec2int(o(:,i)) == -1,
        plot(xx(1),xx(2),'k+');
    end
end
xlabel('x');ylabel('y');
title(['Classification using MLP with dist = ',num2str(dist), ', radius = ',...
       num2str(rad), ' and width = ',num2str(width)]);
% Calculate testing error rate
for i = num_tr+1:num_samp,
    if abs(mysign(o(i)) - nor_data(3,i)) > 1E-6,
        err = err + 1;
    end
end
fprintf('  ------------------------------------\n');
fprintf('   Points tested : %d\n',num_te);
fprintf('    Error points : %d (%5.2f%%)\n',err,(err/num_te)*100);
fprintf('  ------------------------------------\n');
   
fprintf('Mission accomplished!\n');
fprintf('_________________________________________\n');

%%======================= Plot decision boundary ==========================
%% Adding contour to show the boundary
contour(x_b,y_b,z_b,[0 0],'k','Linewidth',1);
%contour(x_b,y_b,z_b,[-1 -1],'k:','Linewidth',2);
%contour(x_b,y_b,z_b,[1 1],'k:','Linewidth',2);
set(gca,'XLim',[xmin xmax],'YLim',[ymin ymax]);


%% That's all, folks
