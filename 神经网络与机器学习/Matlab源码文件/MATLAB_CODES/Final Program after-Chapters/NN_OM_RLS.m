function NN_OM_RLS(dist)
% Exp 4: Classfier using Optimal Manifold and RLS
% offline training
% Trained with K-mean and RLS

% Copyright
% Yanbo Xue
% Adaptive Systems Laboratory
% McMaster University
% yxue@soma.mcmaster.ca
% March 9, 2007
% modified on March 29, 2007 -- Adding new boundary function

%clear;
%%================== Step 0: Generate halfmoon data =======================
rad    = 10;   % central radius of the half moon
width  = 6;    % width of the half moon
%dist   = 1;   % distance between two half moons
num_tr = 1000; % # of training datasets
num_te = 2000; % # of testing datasets
num_samp = num_tr+num_te;% number of samples
fprintf('Radial Basis Function for Classification\n');
fprintf('_________________________________________\n');
fprintf('Generating halfmoon data ...\n');
fprintf('  ------------------------------------\n');
fprintf('  Points generated: %d\n',num_samp);
fprintf('  Halfmoon radius : %2.1f\n',rad);
fprintf('  Halfmoon width  : %2.1f\n',width);
fprintf('      Distance    : %2.1f\n',dist);
fprintf('  ------------------------------------\n');
seed=2e5;
rand('seed',seed);
[data, data_shuffled] = halfmoon(rad,width,dist,num_samp);

% load USPS;
% data_tr = [A4_r(:,1:num_tr/2)  A9_r(:,1:num_tr/2);
%            ones(1,num_tr/2)    -1*ones(1,num_tr/2)];
%   
% data_te = [A4_t(:,1:num_te/2)  A9_t(:,1:num_te/2);
%            ones(1,num_te/2)    -1*ones(1,num_te/2)];
% clear A*;     
% %%===== Shuffling training and testing data ===========
% [n_row, n_col] = size(data_tr);
% shuffle_seq    = randperm(n_col);
% data_shuffled_train = data_tr(:,shuffle_seq);
% [n_row1, n_col1]    = size(data_te);
% shuffle_seq    = randperm(n_col1);
% data_shuffled_test  = data_te(:,shuffle_seq);
% data_shuffled = [data_shuffled_train data_shuffled_test];
% %%====================================================

%%==================== Step 1: Initialization of NN========================
num_hd = 20;   % # of neurons in hidden layer, i.e., # of centroids
num_ou = 1;    % # of neurons in output layer
num_in = 2;    % dimension of input dataset
c_thres = 1E-3; % center changes threshold - if changes are less than this, stop training
c_delta = Inf;  % center changes initialized to be Inf.
w = rand(num_ou,num_hd); % Initial weights

w_thres = 1E-8; % weight changes threshold
mse_thres = 1E-6; % mse threshold
w_delta = Inf;  % weight changes initialized to be Inf.
n  = 1;         % a counter to denote the training progress of K-mean centers
m  = 1;         % a counter to denote the training progress of output weights
eta= 1;      %learning rate for center update
eta_w = annealing(0.6,1E-2,num_tr); % anneal eta
err   = 0;    % a counter to denote the number of error outputs
epochs= 20;
num_train_min = 10;
sig = 0.01;
P = sig^(-1)*eye(num_hd);
lambda = 1;

pos_vector = zeros(1,num_tr); % a vector to denote which point belongs to which center
st = cputime;
%%====== Step 2: Training of K-mean centers - Main loop - offline =========
fprintf('Finding centers using Optimal Manifold method ...\n');
% while c_delta > c_thres && n <= num_tr
%     x = data_shuffled(1:2,n);            % fetch input data
%     %d = myint2vec(data_shuffled(3,n));   % fetch desired response
%     d = data_shuffled(3,n);
%     % calculate Euclidean distance and find position for the min position
%     for i=1:num_hd,
%         eu_dist(i) = distance(x,c(:,i));
%     end
%     [eu_min_value,eu_min_pos] = min(eu_dist);
%     % update corresponding center using this new data
%     c(:,eu_min_pos) = c(:,eu_min_pos) + eta*(x-c(:,eu_min_pos));
%     pos_vector(n) = eu_min_pos;
%     n = n+1;
% end
% 
% %%========== Calculating the variance for each center =====================
% for i = 1 : num_hd,
%     temp_vec = (pos_vector == i); % a logical vector to denote which point 
%                                   % belongs to the ith center
%     dist_temp = 0;
%     for j = 1 : num_tr,
%         if (temp_vec(j)==1),
%             dist_temp = distance(data_shuffled(1:2,j),c(:,i)) + dist_temp;
%         end
%     end
%     sigma(i) = dist_temp/sum(temp_vec);
% end

K_om = num_hd;
lambda_om = 10;
epsilon_om = 0.03;
c = opt_man_new(data_shuffled(1:2,1:num_tr),K_om,lambda_om,epsilon_om);

%%==== Step 3: Calculate spread of K-mean using Eqn. 5.147 of NN book =====
d_max = 0;
for i = 1:num_hd-1,
    if distance(c(:,i),c(:,i+1))>d_max,
        d_max = distance(c(:,i),c(:,i+1));
    else
    end
end
spread = (d_max/sqrt(2*num_hd))*ones(num_hd,1);

%spread = sigma;


fprintf('  ------------------------------------\n');
fprintf('  Number of centers: %d\n',num_hd);
fprintf('        Spread     : %4.1f\n',spread(1));
fprintf('  ------------------------------------\n');

%%======= Step 4: Train the output weights using RLS method ==============
fprintf('Training of output weights using RLS method ...\n');
fprintf('  ------------------------------------\n');

for epoch = 1:epochs,
    %% shuffle the training data for every epoch
    shuffle_seq = randperm(num_tr);
    data_shuffled_tr = data_shuffled(:,shuffle_seq);

    for m = 1:num_tr,
        x = data_shuffled_tr(1:2,m);
        d = data_shuffled(3,m);
        d = data_shuffled_tr(3,m);
        for i = 1:num_hd,
            g(i,:) = exp(-distance(x,c(:,i))^2/(2*spread(i)^2));
        end

        pai = P*g;
        kk = pai/(lambda+g'*pai);
        e = d - w*g;
        w_delta  = (kk*e)';
        ee(m) = d-mysign(w*g);

        if (norm(w_delta) < w_thres) && (m >= num_train_min)
            fprintf('   Weights are stable within %d datasets!\n',m);
            fprintf('   Weights will not be updated for the rest of datasets.\n');
            break;
        end
        w = w + w_delta;
    end
    mse(epoch) = mean(ee.^2);
    fprintf('  Epoch # %d, MSE: %4.2f\n',epoch,mse(epoch));
    if mse(epoch) < mse_thres
        break;
    end
end
fprintf('  Time cost: %4.1f seconds\n',cputime - st);
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
        input = [x_b(x1,y1); y_b(x1,y1)];
        for i = 1 : num_hd
            z_b(x1,y1) = z_b(x1,y1) + w(i)*(exp(-(input-c(:,i))'*(input-c(:,i))/(2*spread(i)^2)));
        end
    end
end

%%============== Adding colormap to the final figure ======================
%figure;
sp = pcolor(x_b,y_b,z_b);
load red_black_colmap;
colormap(red_black);
shading flat;
set(gca,'XLim',[xmin xmax],'YLim',[ymin ymax]);


%%============== Step 5: Testing and plotting result ======================
fprintf('Testing the trained RBF network ...\n');
for i = 1 : num_te;
    x = data_shuffled(1:2,i+num_tr);
    for j = 1:num_hd,
        g(j,:) = exp(-distance(x,c(:,j))^2/(2*spread(j)^2));
    end
    o_test(i) = w*g;
    if o_test(i) > 0,
        plot(x(1),x(2),'rx');
    end
    if o_test(i) < 0,
        plot(x(1),x(2),'k+');
    end
end
xlabel('x');ylabel('y');
title(['Classification using RBF (OM+RLS) with dist = ',num2str(dist), ', radius = ',...
       num2str(rad), ' and width = ',num2str(width)]);
   
% Calculate testing error rate
err = sum(mysign(o_test) - data_shuffled(num_in+1,1+num_tr:num_tr+num_te)~=0);

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

%% Adding K-mean center
%plot(c(1,:),c(2,:),'ow','Linewidth',2,'MarkerSize',10,'MarkerFaceColor','w');
plot(c(1,:),c(2,:),'ow','Linewidth',2,'MarkerSize',10);

%% That's all, folks


function c = opt_man_new(data,K,lambda,epsilon)
%
%% Selecting K Random Data Points from xy_r
%
%  c - center

%data = [x;y];

[n_dim, n_length] = size(data);

N       = n_length;
temp    = randperm(N);
K_index = temp(1:K)';
%
%% Initializing Parameters
%
for i = 1:K,
    gamma(i,:) = data(:,K_index(i));
end

%gamma = gamma';

% N=length(x);
% temp = randperm(N);
% 
% K_index=temp(1:K)';
% %
% %% Initializing Parameters
% %
% gamma(:,1)=x(K_index);
% gamma(:,2)=y(K_index);
P=(1/K)*ones(K,1);
%
%% Iterative Process
%
% iterations=1;
while 1
    for i=1:N
        %exp_dist=exp((-1/lambda)*(((gamma(:,1)-x(i)).^2)+((gamma(:,2)-y(i)).^2)));
        %exp_dist=exp((-1/lambda)*sum(((gamma - repmat([x(i),y(i)],K,1)).^2)')');
        exp_dist=exp((-1/lambda)*sum(((gamma - repmat(data(:,i)',K,1)).^2)')');
        PI = sum(P.*exp_dist);
        P_xi(i,:)=(P/PI ).*exp_dist;
    end
    P=(1/N)*sum(P_xi)';
    gamma_pre=gamma;
    for k=1:K
        %gamma(k,1)=(1/P(k))*(1/N)*sum(x(:).*P_xi(:,k));
        %gamma(k,2)=(1/P(k))*(1/N)*sum(y(:).*P_xi(:,k));
        %gamma(k,:)=(1/P(k)) * (1/N)* (data*P_xi(:,k));
        
        gamma(k,:)=(1/P(k)) * (1/N)* sum((data.*repmat(P_xi(:,k)',n_dim,1))');
    end
    if max(max(abs(gamma-gamma_pre)))<epsilon
        break
    end
%     iterations=iterations+1;
end

c = gamma';
%
% End of Code


