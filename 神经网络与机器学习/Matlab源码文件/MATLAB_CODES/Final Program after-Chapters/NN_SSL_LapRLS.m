function NN_SSL_LapRLS(gamma_I,l,seed)

%clear;
%gamma_I=0.0000;

% genererate 2 moons
r=10;
w=6;
d=-1;
N=1000;
display=0;
%figure(1);
%seed=1e5;
rand('seed',seed);
[X,Y,Xt]=generate_two_moons(r,w,d,N,seed,display);

% pick 2 labeled examples per class
%l=2;
p=find(Y>0); np=length(p); ip=randperm(np);
n=find(Y<0); nn=length(n); in=randperm(nn);

labeled=[p(ip(1:l));n(in(1:l))]
unlabeled=1:length(Y); unlabeled(labeled)=[];


% generate RBF kernel of width 3
options=make_options;
options.Kernel='rbf';
options.KernelParam=3;
K=calckernel(options,X);

% generate Laplacian
options.NN=20; % 20 NN graph
options.GraphWeights='heat';
options.GraphWeightParam='default';
M=laplacian(options,X);

% run LapRLS
y=zeros(length(Y),1);
y(labeled)=Y(labeled);

gamma_A=1e-3;
alpha=laprls(K,y,M,gamma_A,gamma_I);

% plot results
Kt=calckernel(options,X,Xt);
Yt=Kt*alpha;

% for unlabeled data
Xu=X(y==0,:);
Ku=calckernel(options,X,Xu);
Yu=Ku*alpha;
oo = mysign(Yu);

% expected output
for i = 1:length(unlabeled),
    oo_ex(i) = Y(uint64(unlabeled(i)));
end
error = sum(oo' ~= oo_ex);

fprintf('Number of error points: %d; Number of total points: %d\n',error, N);

figure;
plot_toy_problem(X,y,Xt,Yt,oo); %title(['\lambda_I=' num2str(gamma_I) ', Error rate=' num2str(error/N)],'FontSize',10);
xlabel('x');ylabel('y');
title(['LapRLS with dist = ',num2str(d), ...
       ', radius = ', num2str(r), ', \lambda_I=', num2str(gamma_I), ...
       ', error rate = ', num2str(error/N), '(error/N = ', num2str(error), '/',num2str(N), ') and width = ',num2str(w)]);
