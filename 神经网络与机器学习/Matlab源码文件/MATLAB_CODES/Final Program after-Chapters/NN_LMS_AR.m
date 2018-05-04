function NN_LMS_AR(mu)
% Exp: LMS for AR process
% experiment adapted from book Adaptive Filter Theory

% Copyright
% Yanbo Xue
% Adaptive Systems Laboratory
% McMaster University
% yxue@soma.mcmaster.ca
% May 30, 2007

% modified from codes for Adaptive Filter Theory

%%==================== Initialization =====================================
%mu      = 0.02;  
a       = -0.99;   
num_ite = 100;
num_data= 5000;
mult	= 200;
var_v   = 0.019853;
verbose = 0;
sigu2   = 0.93627;  
sigv2   = (1-(a^2)) * sigu2; 
t       = 1 : num_data; 
a2      = a^2; 
p5mu    = 0.5*mu;

%%=================== LMS Main routine ====================================
seed    = 0 : (num_ite-1);
decay   = 0;
Npred   = num_data;
E       = zeros(Npred, num_ite);
WX      = zeros(Npred, num_ite);
Xi0     = 0;

fprintf('LMS for AR process\n');
tt = cputime;
for iter = 1:num_ite,
  randn('seed', seed(iter));
  Xi = filter(1, [1 a], [Xi0 ; sqrt(var_v)*randn(mult*num_data, 1)]);
  if (verbose ~= 0)
      disp(['run # ' num2str(iter)]);
      disp(['  covariance of AR process = ' num2str(cov(Xi))]);%
  end
  Xi = Xi(((mult-1)*num_data+ 2):(mult*num_data+1));
  %lms_AR_pred;
  %%======================= AR pred starts ================================
  Nout = size(Xi, 2);
  % length of maximum number of timesteps that can be predicted
  N = size(Xi, 1);
  % initialize weight matrix and associated parameters for LMS predictor
  W  = zeros(Nout, Nout);
  Wo = [];
  % compute first iteration with Xi(0) = Xi0
  n  = 1;
  Wo = [Wo W];
  xp(n, :) = Xi0 * W';
  e(n, :)  = Xi(n, :) - xp(n, :);
  ne(n) = norm(e(n, :));
  W = W + mu * e(n, : )' * Xi0;
  for n = 2:N,
      % save W matrix
      Wo = [Wo W];
      % predict next sample and error
      xp(n, :) = Xi(n-1, :) * W';
      e(n, :)  = Xi(n, :) - xp(n, :);
      ne(n)    = norm(e(n, :));
      if (verbose ~= 0)
          disp(['time step ', int2str(n), ': mag. pred. err. = ', num2str(ne(n))]);
      end;
      % adapt weight matrix and step size
      W = W + mu * e(n, :)' * Xi(n-1, :);
      if (decay == 1)
          mu = mu  * n/(n+1); % use O(1/n) decay rate
      end;
  end
  %%============================= AR pred ends ============================
  
  E(:,  iter) = e;
  Wx(:, iter) = Wo';
end;
fprintf('Time consumed: %4.2f seconds\n',cputime-tt);

J = sigu2*(1-a2)*(1+p5mu*sigu2) + sigu2*(a2+p5mu*a2*sigu2-0.5*mu*sigu2)*(1-mu*sigu2).^(2*t);

%%========================= Plotting ======================================
figure;
semilogy(J,'r--'); 
hold on;
semilogy(mean((E').^2),'k-');
legend('Theory', 'Experiment')
title(['\eta=',num2str(mu)]);
xlabel('Number of iterations');
ylabel('Mean-square error');