clear;
close all;

%%% parameters
    KernelParam.Type = 2;
    KernelParam.PolyDegree = 2;
    KernelParam.SigmaSq = 0.1;

    LearningParam.LearningRate = 0.05;
    LearningParam.NumofEigenvectors = 3;
    LearningParam.MovementThreshold = 0.00000001;
    LearningParam.IterationLimit = 30000;
%%% parameters

% prepare data
    NumofPatterns = 200;
    TrnPtn = zeros(NumofPatterns, 2);
    for i=1:NumofPatterns
        x = rand(1)*2-1;
        y = (-x)^2+randn(1)/5;
%         y = -x+randn(1)/5;
        TrnPtn(i,1)=x;
        TrnPtn(i,2)=y;
    end
% prepare data

[A, EmpiricalSumKernelMap, EmpiricalKernelSum] = KHAtraining(TrnPtn, KernelParam, LearningParam);

% Contour experiment
% Generate test set: at regular interval
x_test_num = 20;
y_test_num = 20;
test_num = x_test_num*y_test_num;
range = 1;
x_range = -range:(2*range/(x_test_num - 1)):range;
y_offset = 0.5;
y_range = -range+ y_offset:(2*range/(y_test_num - 1)):range+ y_offset;
[xs, ys] = meshgrid(x_range, y_range);
test_patterns(:, 1) = xs(:);
test_patterns(:, 2) = ys(:);
test_features = KHAtesting(EmpiricalSumKernelMap, EmpiricalKernelSum, TrnPtn, test_patterns, A, KernelParam);

% plot it
[NumofEigenvectors,dum] = size(A);

figure(1); clf
for n = 1:NumofEigenvectors,
  subplot(1, NumofEigenvectors, n);
  axis([-range range -range+y_offset range+y_offset]);
  imag = reshape(test_features(:,n), y_test_num, x_test_num);
  axis('xy')
  colormap(gray);
  hold on;
  pcolor(x_range, y_range, imag);
  shading interp
  contour(x_range, y_range, imag, 9, 'w');
  plot(TrnPtn(:,1), TrnPtn(:,2), 'r.')
  hold off;
  axis equal;
  ylim([-0.5 1.5]);
  
end
% plot it
% Contour experiment

