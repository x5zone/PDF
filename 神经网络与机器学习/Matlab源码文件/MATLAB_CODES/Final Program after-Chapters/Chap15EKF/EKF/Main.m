% clc;
% close all;
% clear all;

randn('seed',unidrnd(100)); 

randn('state',sum(100*clock));      %resets it to a different state each time

%==========================================================================
%configure  network
%==========================================================================

global nin nout nhidden1 nhidden2 nwts c1 c2 lambda  r1 r2 r3 R;

%==========================================================================
% define user parameters
%==========================================================================

nexpt = 5;

nep = 260;

nex_tr = 100;

c1 = 1; c2 = 1;         % pars of hyp. tangent fun.

r1 = 3;
r2 = 6;
r3 = 9;

lambda = 1 - 5e-4; 

nin = 2;
nout = 1;
nhidden1 = 5;
nhidden2 = 5;
nwts = nhidden1*(nin+1)+nhidden2*(nhidden1+1)+nout*(nhidden2+1);

%==========================================================================
%Generate input-output data:
%==========================================================================

%load C:\Haran\Research\NN\Code_NN\Ch15_Haykin\Data\seq
generateData;

%==========================================================================
% generate data for testing
%==========================================================================

% x = -1:0.02:1;
% y = -1:0.02:1;

x = -9:0.5:9;
y = -9:0.5:9;

x(1) = []; y(1) = [];
[X,Y] = meshgrid(x,y);
X = X(:);
Y = Y(:);
indata = [X Y]';

idx = [];
for i=1:length(indata)
    d = norm(indata(:,i));
    if d >= r3
        idx = [idx i];
    end;
end;
indata(:,idx) = [];

TrueClass = findTrueClass(indata);

C = zeros(2);

%==========================================================================

tic;

for expt = 1:nexpt
    
    fprintf('=====================================\n');
    fprintf('expt in process = %d\n',expt); 
    fprintf('=====================================\n');
    
    seq = seq(:, randperm(length(seq)));
    
    W1 = (rand(nhidden1,nin+1) - 0.5)/sqrt(nin+1);
    W2 = (rand(nhidden2,nhidden1+1) - 0.5)/sqrt(nhidden1+1);
    W3 = (rand(nout,nhidden2+1) -0.5)/sqrt(nhidden2+1);
    W = [W1(:); W2(:); W3(:)];
    
    %======================================================================
    % train
    %======================================================================
    
    Pkk = 10*eye(nwts);

    R = 1;
    
    nCor = [];
    
    for ep = 1:nep 
    
        %fprintf('epoch in process = %d\n',ep); 
    
        inexArray_tr = [];

        %======================================================================
        % Online (stochastic) learning :
        %======================================================================
        
        for ex = 1:nex_tr 
            
            ptr = ceil(length(seq)*rand);
                        
            inex = seq(1:nin,ptr); 
            
            outex = seq(nin+1:(nin+nout),ptr);
            
            [W,Pkk] = EKF(W,Pkk,inex,outex);    
            
            inexArray_tr = [inexArray_tr inex];
    
        end;
        
        if rem(ep,10) == 0
            
            EstClass = feedfwd(W,indata);
            
            class = (TrueClass >= 0);
            
            classhat = (EstClass >= 0);
            
            tmp = 100*length(find(class == classhat))/size(indata,2);
            
            nCor = [nCor  tmp];            
           
        end;
    
    end;    % epoch    
   
    BignCor(expt,:) = nCor;  
    
   
end; % expt
toc;

%=========================================================================
%Plot training results:
%=========================================================================

EstClass_tr = feedfwd(W,inexArray_tr);

classhat_tr = (EstClass_tr >=0);

gridseq_tr = [inexArray_tr; classhat_tr];

figure;
subplot(121);

plotDecisionBdy(gridseq_tr);
title('Training');

%=========================================================================
% save results
%========================================================================

if (nexpt > 1), 
    nCor_ekf = mean(BignCor);     
else
    nCor_ekf = BignCor;
end;

%save('C:\Haran\Research\NN\Code_NN\Ch15_Haykin\Data\nCor_ekf','nCor_ekf');

gridseq = [indata; classhat];

%save('C:\Haran\Research\NN\Code_NN\Ch15_Haykin\Data\gridseq','gridseq');

%=========================================================================
%Plot test results:
%=========================================================================

subplot(122);
plotDecisionBdy(gridseq);
title('Testing');

figure;
plot(nCor_ekf);
xlabel('Number of  Epochs');
ylabel('Correct Classification (%)');
grid on;

