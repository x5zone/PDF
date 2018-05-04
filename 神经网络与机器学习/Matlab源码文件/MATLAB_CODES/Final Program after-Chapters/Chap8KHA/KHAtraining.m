function [A, EmpiricalSumKernelMap, EmpiricalKernelSum] = KHAtraining(TrnPtn, KernelParam, LearningParam)

% parameters for KHA
    LearningRate = LearningParam.LearningRate;
    NumofEigenvectors = LearningParam.NumofEigenvectors;
    MovementThreshold = LearningParam.MovementThreshold;
    IterationLimit = LearningParam.IterationLimit;
% parameters for KHA

% Initialize A: dual variables
    [NumofPatterns,dum] = size(TrnPtn);
    A = rand(NumofEigenvectors, NumofPatterns);
    A = A*2-1;
    AOld = A;
% Initialize A: dual variables

% Compute empirical 'sum' kernel map for each patterns
    EmpiricalSumKernelMap = zeros(1, NumofPatterns);
    EmpiricalKernelSum = 0;
    for i=1:NumofPatterns
        EmpiricalSumKernelMap(i) = sum(KernelforKHA(TrnPtn(i,:), TrnPtn, KernelParam));
    end
    EmpiricalKernelSum = sum(EmpiricalSumKernelMap);
% Compute empirical sum kernel map for each patterns

% KHA Iteration
    IterationCounter = 0;
    IterationTerminationCondition = -1;
    Movement = 0;
    while IterationTerminationCondition == -1
            for i=1:NumofPatterns
                % Prepare empirical kernel map for test pattern i: to compute sum 1
                        EmpiricalTestKernelMap = KernelforKHA(TrnPtn(i,:), TrnPtn, KernelParam);
                % Prepare empirical kernel map for test pattern i: to compute sum 1
     
                % Compute output Y
                    Y=zeros(1, NumofEigenvectors);
                    OrthogonalizingTerms=zeros(1, NumofPatterns);

                    Sum1 = EmpiricalTestKernelMap*A';
        
                    Sum2 = EmpiricalSumKernelMap*A';
                    Sum2 = Sum2/NumofPatterns;

                    AlphaSum = sum(A');
                    Sum3 = EmpiricalSumKernelMap(i)*AlphaSum;
                    Sum3 = Sum3/NumofPatterns;
        
                    Sum4 = EmpiricalKernelSum*AlphaSum;
                    Sum4 = Sum4/(NumofPatterns^2);

                    Y=Sum1-Sum2-Sum3+Sum4;
                % Compute output Y

                % Update A
                    A = A - LearningRate*(tril((Y'*Y),0)*A);
                    A(:,i) = A(:,i) + LearningRate*Y';
                % Update A
                end
        % Hebbian iteration

        % display status
            if mod(IterationCounter,100) == 0
                Movement = mean(mean(abs(AOld-A))); % dual space distance: should be replaced by the feature space distance(||WOld-W||)
                AOld = A;
                disp(sprintf('Iter %d Movement %g', IterationCounter, Movement));
                pause(0.01);
            end
        % display status

        % update learning rate
            IterationCounter = IterationCounter+1;
%             LearningRate = LearningRate*0.99;
%             if LearningRate < 0.00001
%                 LearningRate = 0.00001;
%             end
        % update learning rate

        % check termination condition
            if Movement < MovementThreshold
                IterationTerminationCondition = 1;
            elseif IterationCounter > IterationLimit
                IterationTerminationCondition = 1;
            end
        % check termination condition
    end
% KHA Iteration

% Normalize eigenvectors in feature space
for i=1:NumofEigenvectors
    AlphaSum = sum(A(i,:));
    FNorm = 0;
    for j=1:NumofPatterns
        for k=1:NumofPatterns
            FNorm = FNorm + (A(i,j)-AlphaSum)*(A(i,k)-AlphaSum)*KernelforKHA(TrnPtn(j,:),TrnPtn(k,:),KernelParam);
        end
    end
    if FNorm ~= 0
        A(i,:) = A(i,:)/sqrt(FNorm);
    end
end
% Normalize eigenvectors in feature space

