%%% ====== function list for NN-LM book ======== 
%%%          For after-chapter problems

clear;
clc;
close all;

%=============== Chapter 1 ============
%%% fig 1.6 (a) (b)
fprintf('Figures of After-the-Chapter Problems ...\n');
fprintf('Reproducing figures of Chapter 1 ...\n');
NN_Perceptron(0);

% Training the perceptron using LMS ...
%   ------------------------------------
%   Points trained : 1000
%        Time cost : 0.66 seconds
%   ------------------------------------
% Testing the perceptron using LMS ...
% Mission accomplished!
%   ------------------------------------
%    Points tested : 2000
%     Error points : 8 ( 0.40%)
%   ------------------------------------



%=============== Chapter 2 ============
%%% fig 2.8
fprintf('Reproducing figures of Chapter 2 ...\n');
NN_LS(0);
%   ------------------------------------
% Testing the classifier using LS ...
% Mission accomplished!
%   ------------------------------------
%    Points tested : 2000
%     Error points : 43 ( 2.15%)
%   ------------------------------------


%%% fig 2.9

NN_LS_regularization(0,0);
NN_LS_regularization(0,0.1);
NN_LS_regularization(0,1);
NN_LS_regularization(0,10);

%============== Chapter 3 ============
%%% fig 3.10
fprintf('Reproducing figures of Chapter 3 ...\n');
NN_LMS_AR(0.002);
NN_LMS_AR(0.01);
NN_LMS_AR(0.02);
% LMS for AR process
% Time consumed: 45.64 seconds
% LMS for AR process
% Time consumed: 53.27 seconds
% LMS for AR process
% Time consumed: 52.00 seconds

%%% fig 3.11
NN_LMS(0);

%    Points tested : 2000
%     Error points : 71 ( 3.55%)

%%% fig 3.12 Plotting learning curves
figure; hold on;
mse1 = NN_LMS_curve(1); 
mse2 = NN_LMS_curve(0); 
mse3 = NN_LMS_curve(-4);
x = 1:length(mse1);
plot(x,mse1,'*-k',x,mse2,'d-k',x,mse3,'o-k');
legend('d = 1','d = 0', 'd = -4');
xlabel('Number of epochs');ylabel('MSE');
hold off;

%============== Chapter 4 ============
%%% fig 4.16 
fprintf('Reproducing figures of Chapter 4 ...\n');
NN_MLP_classifier(0);
% 
%   Epochs conducted: 50
%         Time cost : 6.22 seconds
%   ------------------------------------
% Testing the MLP ...
%   ------------------------------------
%    Points tested : 2000
%     Error points : 0 ( 0.00%)
%   ------------------------------------

%%% fig 4.19(a)(b)
NN_MLP_Lorenz;
%Training is done. It took 10.1 seconds



%============== Chapter 5 ============
%%% fig 5.10 (a) (b)
fprintf('Reproducing figures of Chapter 5 ...\n');
[c1,spread1] = NN_RBF_RLS(1,6);
[c2,spread2] = NN_RBF_RLS(0,6);
[c3,spread3] = NN_RBF_RLS(-1,6);
[c4,spread4] = NN_RBF_RLS(-2,6);
[c5,spread5] = NN_RBF_RLS(-3,6);
[c6,spread6] = NN_RBF_RLS(-4,6);
[c7,spread7] = NN_RBF_RLS(-5,6);
[c8,spread8] = NN_RBF_RLS(-6,6);
figure;
subplot(241); bar(sort(spread1)); title('d=1'); ylim([0,30]);xlim([0,7]);
subplot(242); bar(sort(spread2)); title('d=0'); ylim([0,30]);xlim([0,7]);
subplot(243); bar(sort(spread3)); title('d=-1'); ylim([0,30]);xlim([0,7]);
subplot(244); bar(sort(spread4)); title('d=-2');ylim([0,30]);xlim([0,7]);
subplot(245); bar(sort(spread5)); title('d=-3');ylim([0,30]);xlim([0,7]);
subplot(246); bar(sort(spread6)); title('d=-4');ylim([0,30]);xlim([0,7]);
subplot(247); bar(sort(spread7)); title('d=-5');ylim([0,30]);xlim([0,7]);
subplot(248); bar(sort(spread8)); title('d=-6');ylim([0,30]);xlim([0,7]);

% %====== Results d = 1=================
%  Center coordinates: [ 3.6, 11.0]
%  Center coordinates: [-9.1, 19.0]
%  Center coordinates: [ 4.6,  5.5]
%  Center coordinates: [-8.5,  4.1]
%  Center coordinates: [ 4.0, -4.4]
%  Center coordinates: [11.7, -12.2]
%     Spread(width)  : 13.9
%     Spread(width)  : 11.9
%     Spread(width)  :  4.9
%     Spread(width)  :  8.8
%     Spread(width)  : 19.6
%     Spread(width)  : 18.0
%   ------------------------------------
% Training of output weights using RLS method ...
%   ------------------------------------
%   Time cost:  0.4 seconds
%   ------------------------------------
% Testing the trained RBF network ...
%   ------------------------------------
%    Points tested : 2000
%     Error points : 0 ( 0.00%)
% %====== Results d = 0=================    
%  Center coordinates: [ 4.6, -9.1]
%  Center coordinates: [11.0,  3.6]
%  Center coordinates: [19.0, -4.9]
%  Center coordinates: [11.7,  4.0]
%  Center coordinates: [ 4.1, -7.5]
%  Center coordinates: [-3.4, 11.4]
%     Spread(width)  :  6.3
%     Spread(width)  :  6.2
%     Spread(width)  : 12.5
%     Spread(width)  : 14.1
%     Spread(width)  :  5.5
%     Spread(width)  : 23.1
%   ------------------------------------
% Training of output weights using RLS method ...
%   ------------------------------------
%   Time cost:  2.9 seconds
%   ------------------------------------
% Testing the trained RBF network ...
%   ------------------------------------
%    Points tested : 2000
%     Error points : 2 ( 0.10%)
% %====== Results d = -1=================
%  Center coordinates: [ 5.5, 11.0]
%  Center coordinates: [-9.1, 19.0]
%  Center coordinates: [ 4.6,  3.6]
%  Center coordinates: [-10.2,  4.1]
%  Center coordinates: [ 4.0, -2.4]
%  Center coordinates: [11.7, -6.5]
%     Spread(width)  : 14.7
%     Spread(width)  : 13.0
%     Spread(width)  : 17.3
%     Spread(width)  :  8.4
%     Spread(width)  : 12.0
%     Spread(width)  : 12.7
%   ------------------------------------
% Training of output weights using RLS method ...
%   ------------------------------------
%   Time cost:  2.9 seconds
%   ------------------------------------
% Testing the trained RBF network ...
%   ------------------------------------
%    Points tested : 2000
%     Error points : 20 ( 1.00%)
% %====== Results d = -2=================
%  Center coordinates: [ 3.6,  5.5]
%  Center coordinates: [19.0, 11.0]
%  Center coordinates: [ 4.6, -9.1]
%  Center coordinates: [-5.5, -9.2]
%  Center coordinates: [-1.4,  4.1]
%  Center coordinates: [11.7,  4.0]
%     Spread(width)  : 13.9
%     Spread(width)  : 11.3
%     Spread(width)  : 12.4
%     Spread(width)  : 12.1
%     Spread(width)  : 14.9
%     Spread(width)  : 20.4
%   ------------------------------------
% Training of output weights using RLS method ...
%   ------------------------------------
%   Time cost:  0.5 seconds
%   ------------------------------------
% Testing the trained RBF network ...
%   ------------------------------------
%    Points tested : 2000
%     Error points : 5 ( 0.25%)
% %====== Results d = -3=================    
% Center coordinates: [ 4.6, 11.0]
%  Center coordinates: [19.0, 11.7]
%  Center coordinates: [ 3.6, -9.1]
%  Center coordinates: [11.7,  4.1]
%  Center coordinates: [-0.4, -5.7]
%  Center coordinates: [-4.5,  4.0]
%     Spread(width)  : 12.2
%     Spread(width)  :  8.8
%     Spread(width)  : 10.2
%     Spread(width)  : 19.0
%     Spread(width)  : 10.5
%     Spread(width)  : 10.0
%   ------------------------------------
% Training of output weights using RLS method ...
%   ------------------------------------
%   Time cost:  3.0 seconds
%   ------------------------------------
% Testing the trained RBF network ...
%   ------------------------------------
%    Points tested : 2000
%     Error points : 53 ( 2.65%)
% %====== Results d = -4=================
%  Center coordinates: [11.7,  4.6]
%  Center coordinates: [11.0,  3.6]
%  Center coordinates: [-9.1, 19.0]
%  Center coordinates: [-4.7, 11.7]
%  Center coordinates: [ 4.1, -3.5]
%  Center coordinates: [ 4.0,  0.6]
%     Spread(width)  : 12.6
%     Spread(width)  : 15.6
%     Spread(width)  : 10.0
%     Spread(width)  : 12.1
%     Spread(width)  : 14.0
%     Spread(width)  : 19.1
%   ------------------------------------
% Training of output weights using RLS method ...
%   ------------------------------------
%   Time cost:  3.0 seconds
%   ------------------------------------
% Testing the trained RBF network ...
%   ------------------------------------
%    Points tested : 2000
%     Error points : 69 ( 3.45%)
% %====== Results d = -5=================
% Center coordinates: [11.7,  3.6]
%  Center coordinates: [ 4.6, 11.0]
%  Center coordinates: [-9.1, 19.0]
%  Center coordinates: [-3.7, -2.5]
%  Center coordinates: [11.7,  4.1]
%  Center coordinates: [ 4.0,  1.6]
%     Spread(width)  :  9.9
%     Spread(width)  :  7.6
%     Spread(width)  : 14.5
%     Spread(width)  : 11.7
%     Spread(width)  : 19.0
%     Spread(width)  : 21.1
%   ------------------------------------
% Training of output weights using RLS method ...
%   ------------------------------------
%   Time cost:  2.8 seconds
%   ------------------------------------
% Testing the trained RBF network ...
%   ------------------------------------
%    Points tested : 2000
%     Error points : 101 ( 5.05%)
% %====== Results d = -6=================    
%  Center coordinates: [11.0, -4.9]
%  Center coordinates: [ 3.6, -9.1]
%  Center coordinates: [ 4.6, 19.0]
%  Center coordinates: [ 4.1, 11.4]
%  Center coordinates: [-1.5,  4.0]
%  Center coordinates: [11.7,  2.6]
%     Spread(width)  :  8.4
%     Spread(width)  :  8.6
%     Spread(width)  :  7.9
%     Spread(width)  : 20.1
%     Spread(width)  : 11.8
%     Spread(width)  :  8.9
%   ------------------------------------
% Training of output weights using RLS method ...
%   ------------------------------------
%   Time cost:  3.0 seconds
%   ------------------------------------
% Testing the trained RBF network ...
%   ------------------------------------
%    Points tested : 2000
%     Error points : 92 ( 4.60%)

%%%% fig 5.11
NN_RBF_LMS(-5,20);
%    Points tested : 2000
%     Error points : 11 ( 0.55%)
NN_RBF_LMS(-6,20);
%    Points tested : 2000
%     Error points : 17 ( 0.85%)

%============== Chapter 6 ============
%%% fig 6.24 (a) (b)
fprintf('Reproducing figures of Chapter 6 ...\n');
C = 10.^(0:0.25:10);
for i = 1: length(C);
    err(i) = NN_SVM(-6.5,C(i));
end
figure;
loglog(C,err/2000,'Color','k','Marker','o','LineWidth',2);
grid on;
hold on;
for i = 1: length(C);
    err(i) = NN_SVM(-6.75,C(i));
end
%figure;

loglog(C,err/2000,'Color','r','Marker','o','LineWidth',2);
xlabel('C');
ylabel('Error rate');
title('Pattern classification using SVM');
legend('d = -6.5','d = -6.75');


%%% fig 6.25
NN_SVM_twist_fist(500);
% Error points : 14 ( 7.00%)
NN_SVM_twist_fist(100);
%    Error points : 16 ( 8.00%)
NN_SVM_twist_fist(2500);
%  Error points : 15 ( 7.50%)

%============== Chapter 7 ============
%%% fig 7.20(a)
fprintf('Reproducing figures of Chapter 7 ...\n');
NN_SSL_LapRLS(0.1,1,1e5);
NN_SSL_LapRLS(0.1,1,2e5);
NN_SSL_LapRLS(0.1,1,3e5);
%%% fig 7.20(b)
NN_SSL_LapRLS(0.1,2,1e5);
NN_SSL_LapRLS(0.1,2,2e5);
NN_SSL_LapRLS(0.1,2,3e5);

%============== Chapter 8 ============
%%% fig 8.17(a)
fprintf('Reproducing figures of Chapter 8 ...\n');
[W_out, mval_other] = NN_GHA('lena.tif');
%%% fig 8.17(b)
W = NN_GHA_A('pepper.tif',W_out, mval_other);
%%% fig 8.18
wd = cd;
cd([wd,'\Chap8KHA']);
ToyExampleforKHA;
cd ..;


%============== Chapter 10 ============
%%% fig 10.31
fprintf('Reproducing figures of Chapter 10 ...\n');
NN_OM_RLS(-6);
%    Points tested : 2000
%     Error points : 12 ( 0.60%)

%============== Chapter 14 ============
wd = cd;
fprintf('Reproducing figures of Chapter 14 ...\n');
cd([wd,'\Chap14Bayesian']);
particle_filter;
cd ..;

%============ Chapter 15 ================
wd = cd;
fprintf('Reproducing figures of Chapter 15 ...\n');
cd([wd,'\Chap15EKF\EKF']);
Main;
cd ..;
cd ..;