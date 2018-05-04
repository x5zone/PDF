function NN_RBF_LMS(dist,num_hd)
% Exp 4: Classfier using Radial Basis Function (RBF) network
% offline training
% Trained with K-mean and LMS

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

%%==================== Step 1: Initialization of NN========================
%num_hd = 20;   % # of neurons in hidden layer, i.e., # of centroids
num_ou = 1;    % # of neurons in output layer
num_in = 2;    % dimension of input dataset
[data_ini,data_shuffled_ini] = halfmoon(rad,width,dist,num_hd);
c = data_shuffled_ini(1:2,:); % generate initial centers using self-organized method, dim: 2 x num_hd
c_thres = 1E-3; % center changes threshold - if changes are less than this, stop training
c_delta = Inf;  % center changes initialized to be Inf.
w = rand(num_ou,num_hd)./2  - 0.25; % Initial weights

w_thres = 1E-8; % weight changes threshold
mse_thres = 1E-6; % mse threshold
w_delta = Inf;  % weight changes initialized to be Inf.
n  = 1;         % a counter to denote the training progress of K-mean centers
m  = 1;         % a counter to denote the training progress of output weights
eta= 1;      %learning rate for center update
eta_w = annealing(0.6,1E-2,num_tr); % anneal eta
err   = 0;    % a counter to denote the number of error outputs
epochs= 50;
num_train_min = 10;

pos_vector = zeros(1,num_tr); % a vector to denote which point belongs to which center
st = cputime;
%%====== Step 2: Training of K-mean centers - Main loop - offline =========
fprintf('Finding centers using K-mean method ...\n');
while c_delta > c_thres && n <= num_tr
    x = data_shuffled(1:2,n);            % fetch input data
    %d = myint2vec(data_shuffled(3,n));   % fetch desired response
    d = data_shuffled(3,n);
    % calculate Euclidean distance and find position for the min position
    for i=1:num_hd,
        eu_dist(i) = distance(x,c(:,i));
    end
    [eu_min_value,eu_min_pos] = min(eu_dist);
    % update corresponding center using this new data
    c(:,eu_min_pos) = c(:,eu_min_pos) + eta*(x-c(:,eu_min_pos));
    pos_vector(n) = eu_min_pos;
    n = n+1;
end

%%========== Calculating the variance for each center =====================
for i = 1 : num_hd,
    temp_vec = (pos_vector == i); % a logical vector to denote which point 
                                  % belongs to the ith center
    dist_temp = 0;
    for j = 1 : num_tr,
        if (temp_vec(j)==1),
            dist_temp = distance(data_shuffled(1:2,j),c(:,i)) + dist_temp;
        end
    end
    sigma(i) = dist_temp/sum(temp_vec);
end


% %%==== Step 3: Calculate spread of K-mean using Eqn. 5.147 of NN book =====
% d_max = 0;
% for i = 1:num_hd-1,
%     if distance(c(:,i),c(:,i+1))>d_max,
%         d_max = distance(c(:,i),c(:,i+1));
%     else
%     end
% end
% spread = (d_max/sqrt(2*num_hd))*ones(num_hd,1);

spread = sigma;


fprintf('  ------------------------------------\n');
fprintf('  Number of centers: %d\n',num_hd);
fprintf('        Spread     : %4.1f\n',spread);
fprintf('  ------------------------------------\n');

%%======= Step 4: Train the output weights using LMS method ==============
fprintf('Training of output weights using LMS method ...\n');
fprintf('  ------------------------------------\n');

for epoch = 1:epochs,
    %% shuffle the training data for every epoch
    %[n_row, n_col] = size(data_shuffled);
    shuffle_seq = randperm(num_tr);
    data_shuffled_tr = data_shuffled(:,shuffle_seq);

    for m = 1:num_tr,
        x = data_shuffled_tr(1:2,m);
        %d = myint2vec(data_shuffled(3,m));
        d = data_shuffled_tr(3,m);
        for i = 1:num_hd,
            g(i,:) = exp(-(x-c(:,i))'*(x-c(:,i))/(2*spread(i)^2));
        end
        %g = exp(-(dot(x*ones(1,num_hd)-c,x*ones(1,num_hd)-c).^2)'./(2*spread.^2)); % output of hidden layer
        %o(m) = myvec2int(w*g);
        o(m) = w*g;
        e = d - w*g;
        w_delta = eta_w(m)*e*g';
        ee(m) = e;

        if (norm(w_delta) < w_thres) && (m >= num_train_min)
            fprintf('   Weights are stable within %d datasets!\n',m);
            fprintf('   Weights will not be updated for the rest of datasets.\n');
            break;
        end
        w = w + w_delta;
        %     if n == num_tr,
        %         fprintf('   Datasets are not enough for the NN to get stablized.\n');
        %         return;
        %     end
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
%        z_b(x1,y1) = w*exp(-(dot(input*ones(1,num_hd)-c,input*ones(1,num_hd)-c).^2)'./(2*spread.^2));
        for i = 1 : num_hd
            z_b(x1,y1) = z_b(x1,y1) + w(i)*(exp(-(input-c(:,i))'*(input-c(:,i))/(2*spread(i)^2)));
        end
    end
    %waitbar((x1)/size(x,1),wh)
    %set(wh,'name',['Progress = ' sprintf('%2.1f',(x1)/size(x,1)*100) '%']);
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
        g(j,:) = exp(-(x-c(:,j))'*(x-c(:,j))/(2*spread(j)^2));
    end
    %o(i) = mysign(w*g);
    o(i) = w*g;
    if o(i) > 0,
        plot(x(1),x(2),'rx');
    end
    if o(i) < 0,
        plot(x(1),x(2),'k+');
    end
end
xlabel('x');ylabel('y');
title(['Classification using RBF with dist = ',num2str(dist), ', radius = ',...
       num2str(rad), ' and width = ',num2str(width)]);
% Calculate testing error rate
for i = 1:num_te,
    if abs(mysign(o(i)) - data_shuffled(3,i+num_tr)) > 1E-6,
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

%% Adding K-mean center
%plot(c(1,:),c(2,:),'ow','Linewidth',2,'MarkerSize',10,'MarkerFaceColor','w');
plot(c(1,:),c(2,:),'ow','Linewidth',2,'MarkerSize',10);

%% That's all, folks


