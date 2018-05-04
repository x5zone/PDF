% Computer Experiment: Particle Filter
% In this problem we use a particle filter to solve a nonlinear tracking
% problem in computer vision.

clear;
%close all;

% define parameters
N = 100;                       % number of particles
bar_width = 10;                % width of foreground bars
var_measurement = [2 2 2 2];   % measurement noise variance
var_process = [2 2 8 8];       % process noise variance

% define constants
D = 4;                         % state space dimension
T = 150;                       % simulation time
object_color = [0.5 0.5 0.5];  % color of the tracked object

% initialize random number generator
rand('state',430896);          % constant value to get the same results  
randn('state',430896);         % use 'sum(100*clock)' for varying results

% initialize variables
background = simulation(0, object_color, bar_width);
[height, width, radio] = size(background);
R = diag(var_measurement);
Q = diag(var_process);
xPFmean = zeros(D,T);
yPFmean = zeros(D,T);
xReal = zeros(T);
yReal = zeros(T);
xparticle_pf = ones(D,T,N);
xparticlePred_pf = ones(D,T,N);  
yPred_pf = ones(D,T,N);          
w = ones(T,N);                   

% initialize particle states
for i=1:N,
  xparticle_pf(1:2,1,i) = [50; 150];
  xparticle_pf(3:4,1,i) = randn(2,1);
  y(1:2,1) = [50; 150];
  y(3:4,1) = [0; 0];
  xPFmean(1:2,1) = [50; 150];
  yPFmean(1:2,1) = [50; 150];
end;

% initialize depth evaluation
edges = edge(rgb2gray(background),'canny');
st = strel('square',5);
edges = imclose(edges, st);
bins = double(zeros(height,width));

for t=2:T,    
  % load the current frame
  [im, xReal(t), yReal(t)] = simulation((t-2)/T, object_color, bar_width);
  
  % get the current observation for calculating the likelihood
  [y(:,t) observable] = observe(im, y(:,t-1), object_color);
    
  for i = 1:N,
    % sample from prior density
    xparticlePred_pf(:,t,i) = feval('process_model',...
        xparticle_pf(:,t-1,i)) + sqrtm(Q)*randn(D,1); 
    % get predicted observation, equals the current state here
    yPred_pf(:,t,i) = xparticlePred_pf(:,t,i);
    
    % check if predicted observation is outside bounds
    if(yPred_pf(1,t,i) < 1 | yPred_pf(1,t,i) > width |...
            yPred_pf(2,t,i) < 1 | yPred_pf(2,t,i) > height) 
      w(t,i) = 1e-99;
    
    % check if object is observable
    elseif(observable == false)
      w(t,i) = 1/N;
      
    % calculate likelihood from observation and predicted observation
    else
      w(t,i) = exp(-0.5*(y(:,t)-yPred_pf(:,t,i))'*inv(R)*...
          (y(:,t)-yPred_pf(:,t,i))) + 1e-99;
    end
  end
  
  % normalize the weights
  w(t,:) = w(t,:)./sum(w(t,:)); 
  
  % apply systematic resampling algorithm
  new_index = resampling(1:N,w(t,:)');
  xparticle_pf(:,t,1:N) = xparticlePred_pf(:,t,new_index);
  
  % compute posterior mean predictions
  for d=1:D,
    xPFmean(d,t) = mean(xparticle_pf(d,t,1:N));
    yPFmean(d,t) = mean(yPred_pf(d,t,1:N));
  end
  
  % compute error (for evaluation only)
  mse(t) = norm([xReal(t);yReal(t)] - xPFmean(1:2,t));  
  
  % depth evaluation (foreground/background)
  filled = imfill(edges,floor([yPFmean(2,t), yPFmean(1,t)]),4);
  area = filled-edges;
  normalization = norm(area);
  if(normalization ~= 0)
    area = area./normalization;
  end
  
  % update depth map by checking the observability of the object
  if(observable == true)
      bins = bins + area;
  else
      bins = bins - area;
  end 
  
  % visualize particle map
  figure(1), subplot(1,2,1);
  imshow(im);
  hold on;
  for i=1:N,
      % position of the particle
      plot(xparticle_pf(1,t,i), xparticle_pf(2,t,i),'ro'); 
      % velocity of the particle
      plot([xparticle_pf(1,t,i);xparticle_pf(1,t,i)+...
          xparticle_pf(3,t,i)], [xparticle_pf(2,t,i);...
          xparticle_pf(2,t,i)+xparticle_pf(4,t,i)],'r-'); 
  end  
  % mean position
  plot(xPFmean(1,t), xPFmean(2,t), 'r*');
  % mean velocity
  plot([xPFmean(1,t); xPFmean(1,t)+xPFmean(3,t)], [xPFmean(2,t);...
      xPFmean(2,t)+xPFmean(4,t)], 'r-'); 
  hold off;
  subplot(1,2,2), imshow(bins,[]);
  pause(0.1);
end

% plot trajectories
figure(1);
clf;
imshow(im);
hold on;
plot(yPFmean(1,:), yPFmean(2,:), 'b-');
plot(y(1,:), y(2,:), 'rx');
hold off;

% plot depth map
figure(2), imshow(bins,[]);

% plot performance measure
figure(3), plot(1:T, mse);
xlabel('Time');
title('MSE between real and estimated object coordinates');