function d = KHAtesting(EmpiricalSumKernelMap, EmpiricalKernelSum, TrnPtn, TestPtn, A, KernelParam)

[NumofTrnPatterns,dum] = size(TrnPtn);
[NumofTestPatterns,dum] = size(TestPtn);
[NumofEigenvectors,dum] = size(A);

for i=1:NumofTestPatterns
    % Prepare empirical kernel map for test pattern i: to compute sum 1
        EmpiricalTestKernelMap = KernelforKHA(TestPtn(i,:), TrnPtn, KernelParam);
    % Prepare empirical kernel map for test pattern i: to compute sum 1
     
    % Compute output Y
        Y=zeros(1, NumofEigenvectors);

        Sum1 = EmpiricalTestKernelMap*A';
        
        Sum2 = EmpiricalSumKernelMap*A';
        Sum2 = Sum2/NumofTrnPatterns;
        
        AlphaSum = sum(A');
        Sum3 = sum(EmpiricalTestKernelMap)*AlphaSum;
        Sum3 = Sum3/NumofTrnPatterns;
        
        Sum4 = EmpiricalKernelSum*AlphaSum;
        Sum4 = Sum4/(NumofTrnPatterns^2);

        Y=Sum1-Sum2-Sum3+Sum4;
    % Compute output Y
    d(i,:) = Y;
end
