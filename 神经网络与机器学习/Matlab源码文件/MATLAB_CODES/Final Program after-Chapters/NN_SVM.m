function err = NN_SVM(dist,C)%[nsv, alpha, b0, data] = NN_SVM(dist,C) %(X,Y,C)
%NN_SVM: Support Vector Machine for Classification
%
%  Usage: [nsv alpha bias] = NN_SVM(X,Y,ker,C)
%
%  Parameters: X      - Training inputs
%              Y      - Training targets
%              ker    - kernel function
%              C      - upper bound (non-separable case)
%              nsv    - number of support vectors
%              alpha  - Lagrange Multipliers
%              b0     - bias term
%
%  Original code: Steve Gunn (srg@ecs.soton.ac.uk)
%  Modified by Yanbo Xue (yxue@soma.mcmaster.ca) for non-commercial use
%  March 28, 2007
% 
%%--------------------- BEGIN OF MAIN ROUTINE -----------------------------

%%============== Step 0A: Generate halfmoon data ==========================
rad      = 10;   % central radius of the half moon
width    = 6;    % width of the half moon
%dist     = -4;   % distance between two half moons
num_tr   = 300;  % # of training datasets
num_te   = 2000; % # of testing datasets
num_samp = num_tr + num_te;% number of samples
fprintf('Support Vector Machine for Classification\n');
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
warning off;

%%================== Step 0B: Initialization of SVM =======================
%C      = 25;  % upper bound (non-separable case)
p1     = 1;    % sigma for RBF, width
epsilon= 1e-5; % threshold for Support Vector Detection
err    = 0;    % a counter to denote the number of error outputs
b0     = 0;    % Implicit bias, 0 for RBF kernal

%%==== Step 1: Preprocess the input data, remove mean and normalize =======
mean0 = [mean(data(1:2,:)')';0];         % mean of the original data
for i = 1:num_samp,
    data_norm(:,i) = data_shuffled(:,i) - mean0;
end
max0  = [max(abs(data_norm(1:2,:)'))';1];% max of the original data
for i = 1:num_samp,
    data_norm(:,i) = data_norm(:,i)./max0;
end

%%========== Step 2: Fetch Training and Testing Data ======================
X_tr = data_norm(1:2,1:num_tr)';
Y_tr = data_norm(3,1:num_tr)';
X_te = data_norm(1:2,1+num_tr:num_tr+num_te)';
Y_te = data_norm(3,1+num_tr:num_tr+num_te)';

%==================== Step 3: Construct the Kernel matrix =================
fprintf('Constructing RBF kernal matrix ...\n');
H = zeros(num_tr,num_tr);
for i=1:num_tr
    for j=1:num_tr
        H(i,j) = Y_tr(i)*Y_tr(j)*(exp(-(X_tr(i,:)-X_tr(j,:))*(X_tr(i,:)-X_tr(j,:))'/(2*p1^2)));
    end
end
c = -ones(num_tr,1);
H = H + 1E-10*randn(size(H)); % add artifical noise to avoid Hessian problem.

%============ Step 4: Use QP to solve the Optimization Problem ============
% Set up the parameters for the Optimization problem
fprintf('Optimizing using QP ...\n');
vlb = zeros(num_tr,1); % Set the bounds: alphas >= 0
vub = C*ones(num_tr,1);%                 alphas <= C
x0  = zeros(num_tr,1); % The starting point is [0 0 0   0]
neqcstr = 0;           % Set the number of equality constraints (1 or 0)
A = []; 
b = [];
% Running QP to solve the optimization problem
% The QP is implemented using C++ and then exported to a .dll file
st = cputime;
[alpha lambda how] = new_qp(H, c, A, b, vlb, vub, x0, neqcstr);

%======== Step 5: Compute the number of Support Vectors ===================
svi = find( alpha > epsilon);
nsv = length(svi);

fprintf('  ------------------------------------\n');
fprintf('   Points trained: %d\n',num_tr);
fprintf('   Execution time: %4.1f seconds\n',cputime - st);
fprintf('       Status    : %s\n',how);
fprintf('         C       : %f\n',C);
w2 = alpha'*H*alpha;
fprintf('       |w0|^2    : %f\n',w2);
fprintf('       Margin    : %f\n',2/sqrt(w2));
fprintf('      Sum alpha  : %f\n',sum(alpha));
fprintf('  Support Vectors: %d (%3.1f%%)\n',nsv,100*nsv/num_tr);
fprintf('  ------------------------------------\n');

% %============= Step 6: Plot training result ===============================
% figure;
% svcplot_train(X_tr,Y_tr,alpha,b0,epsilon,p1,mean0,max0);
% xlabel('x');ylabel('y');
% title(['Classification using SVM with dist = ',num2str(dist),', radius = ',...
%        num2str(rad), ' and width = ',num2str(width), '(Training)']);

%%=============== Step 7: Test the trained SVM ============================
fprintf('Testing the trained SVM ... \n');
H_te = zeros(num_te,num_tr);
for i=1:num_te
    for j=1:num_tr
        H_te(i,j) = Y_tr(j)*(exp(-(X_te(i,:)-X_tr(j,:))*(X_te(i,:)-X_tr(j,:))'/(2*p1^2)));
    end
end
Y_pred = sign(H_te*alpha + b0);
% Calculate testing error rate
for i = 1:num_te,
    if abs(Y_pred(i)-Y_te(i)) > 1E-6,
        err = err + 1;
    end
end
fprintf('  ------------------------------------\n');
fprintf('   Points tested : %d\n',num_te);
fprintf('    Error points : %d (%5.2f%%)\n',err,(err/num_te)*100);
fprintf('  ------------------------------------\n');

% %%===================== Step 8: Plot testing result =======================
% figure;
% svcplot_test(X_tr,Y_tr,X_te,Y_pred,alpha,b0,epsilon,p1,mean0,max0);
% xlabel('x');ylabel('y');
% title(['Classification using SVM with d = ',num2str(dist), ', r = ',...
%        num2str(rad), ' and width = ',num2str(width), '( C = ',num2str(C),' Testing)']);
%    
% fprintf('Mission accomplished!\n');
% fprintf('_________________________________________\n');

%================== Final: Output data for further process ================
data.X_tr = X_tr; % training input
data.Y_tr = Y_tr; % training output
data.X_te = X_te; % testing input
data.Y_te = Y_te; % ideal testing output
data.Y_pred=Y_pred;% predicted testing output
data.mean = mean0;% mean value of the data (training + testing)
data.max  = max0; % max value of the data (training + testing)
data.dist = dist; % parameter of halfmoon data: distance
data.rad  = rad;  % parameter of halfmoon data: radius
data.width= width;% parameter of halfmoon data: width
data.err  = err;  % number of error outputs

%%-------------------- END OF MAIN ROUTINE --------------------------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                                                       %%
%%          Two subroutines to plot the training and testing result      %%
%%                                                                       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [h] = svcplot_train(X_tr,Y_tr,alpha,bias,epsilon,p1,mean0,max0)
%svcplot_train: Support Vector Machine Plotting Routine for Training data
%
%  Usage: svcplot_train(X_tr,Y_tr,alpha,bias,epsilon,p1,mean0,max0)
%
%  Parameters: X_tr   - Training inputs
%              Y_tr   - Training outputs
%              alpha  - Lagrange Multipliers
%              bias   - Bias term
%            epsilon  - Threshold for support vector detection
%               p1    - Sigma for RBF, spread width
%              mean0  - Mean value of simulation data
%              max0   - Max value of simulation data

hold on;

%%================== Plot Training Points =================================
for i = 1:size(Y_tr)
    XX = X_tr(i,1)*max0(1) + mean0(1); % scale x back to original
    YY = X_tr(i,2)*max0(2) + mean0(2); % scale y back to original
    if (Y_tr(i) == 1)
        h(1) = plot(XX,YY,'r+','LineWidth',1); % Class A
    end
    if (Y_tr(i) == -1)
        h(2) = plot(XX,YY,'kx','LineWidth',1); % Class B
    end
    if ((abs(alpha(i)) > epsilon)&&Y_tr(i)==1)
        plot(XX,YY,'ro','Linewidth',2); % Support Vector
    end
    if ((abs(alpha(i)) > epsilon)&&Y_tr(i)==-1)
        plot(XX,YY,'ko','Linewidth',2); % Support Vector
    end
end

%%================= Plot Decision Boundary ================================
xmin = min(X_tr(:,1));
xmax = max(X_tr(:,1));
ymin = min(X_tr(:,2));
ymax = max(X_tr(:,2));
[x,y]= meshgrid(xmin:(xmax-xmin)/50:xmax,ymin:(ymax-ymin)/50:ymax);
z    = bias*ones(size(x));
%wh = waitbar(0,'Plotting...');
for x1 = 1 : size(x,1)
    for y1 = 1 : size(x,2)
        input(1) = x(x1,y1);
        input(2) = y(x1,y1);
        for i = 1 : length(Y_tr)
            if (abs(alpha(i)) > epsilon)
                z(x1,y1) = z(x1,y1) + Y_tr(i)*alpha(i)*(exp(-(input-X_tr(i,:))*(input-X_tr(i,:))'/(2*p1^2)));
            end
        end
    end
    %waitbar((x1)/size(x,1),wh)
    %set(wh,'name',['Progress = ' sprintf('%2.1f',(x1)/size(x,1)*100) '%']);
end
%close(wh);

% Scale data points back to original value
x = x*max0(1) + mean0(1);
y = y*max0(2) + mean0(2);
xmin = xmin*max0(1) + mean0(1); % scale xmin
xmax = xmax*max0(1) + mean0(1); % scale xmax
ymin = ymin*max0(2) + mean0(2); % scale ymin
ymax = ymax*max0(2) + mean0(2); % scale ymax

set(gca,'XLim',[xmin xmax],'YLim',[ymin ymax]);

% Plot Boundary contour
contour(x,y,z,[0 0],'k','Linewidth',1);
contour(x,y,z,[-1 -1],'k:','Linewidth',1);
contour(x,y,z,[1 1],'k:','Linewidth',1);
hold off
axis on
%set(gca,'XTickLabelMode','auto');
%set(gca,'YTickLabelMode','auto');
return

function svcplot_test(X_tr,Y_tr,X_te,Y_pred,alpha,bias,epsilon,p1,mean0,max0)
%svcplot_test: Support Vector Machine Plotting Routine for Testing data
%
%Usage: svcplot_test(X_tr,Y_tr,X_te,Y_pred,alpha,bias,epsilon,p1,mean0,max0)
%
%  Parameters: X_tr   - Training inputs
%              Y_tr   - Training outputs
%              X_te   - Testing inputs
%              Y_pred - Testing outputs
%              alpha  - Lagrange Multipliers
%              bias   - Bias term
%            epsilon  - Threshold for support vector detection
%               p1    - Sigma for RBF, spread width
%              mean0  - Mean value of simulation data
%              max0   - Max value of simulation data

%%================= Colormaping the figure here ===========================
%%=== In order to avoid the display problem of eps file in LaTeX. =========
hold on;
xmin = min(X_te(:,1));
xmax = max(X_te(:,1));
ymin = min(X_te(:,2));
ymax = max(X_te(:,2));
[x,y]= meshgrid(xmin:(xmax-xmin)/50:xmax,ymin:(ymax-ymin)/50:ymax);
z    = bias*ones(size(x));
%wh = waitbar(0,'Plotting testing result...');
for x1 = 1 : size(x,1)
    for y1 = 1 : size(x,2)
        input(1) = x(x1,y1);, input(2) = y(x1,y1);
        for i = 1 : length(Y_tr)
            if (abs(alpha(i)) > epsilon)
                z(x1,y1) = z(x1,y1) + Y_tr(i)*alpha(i)*(exp(-(input-X_tr(i,:))*(input-X_tr(i,:))'/(2*p1^2)));
            end
        end
    end
    %waitbar((x1)/size(x,1),wh)
    %set(wh,'name',['Progress = ' sprintf('%2.1f',(x1)/size(x,1)*100) '%']);
end
%close(wh);

x = x*max0(1) + mean0(1); % scale x back to original
y = y*max0(2) + mean0(2); % scale y back to original
xmin = xmin*max0(1) + mean0(1); % scale xmin
xmax = xmax*max0(1) + mean0(1); % scale xmax
ymin = ymin*max0(2) + mean0(2); % scale ymin
ymax = ymax*max0(2) + mean0(2); % scale ymax

sp = pcolor(x,y,z);
load red_black_colmap;
colormap(red_black);
shading flat;
set(gca,'XLim',[xmin xmax],'YLim',[ymin ymax]);
num_te = size(X_te,1);

for i = 1:num_te,
    XX = X_te(i,1)*max0(1) + mean0(1); % scale x back to original
    YY = X_te(i,2)*max0(2) + mean0(2); % scale y back to original
    if Y_pred(i) == 1,
        plot(XX,YY,'r+');
    else
        plot(XX,YY,'kx');
    end
end

%%======================= Plot decision boundary ==========================
contour(x,y,z,[0 0],'k','Linewidth',1);
contour(x,y,z,[-1 -1],'k:','Linewidth',1);
contour(x,y,z,[1 1],'k:','Linewidth',1);
% set(gca,'XTickLabelMode','auto');
% set(gca,'YTickLabelMode','auto');
hold off
axis on
return


