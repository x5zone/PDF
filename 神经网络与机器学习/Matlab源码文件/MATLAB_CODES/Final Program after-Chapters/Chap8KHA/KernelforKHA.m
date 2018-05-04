% Kernel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = KernelforKHA(x1, x2, KernelParam)
switch KernelParam.Type
    case 1 % linear
        y = LinearKernel(x1, x2);
    case 2 % polynomial
        y = PolynomialKernel(x1, x2, KernelParam.PolyDegree);
    case 3 % Gaussian
        y = GaussianKernel(x1, x2, KernelParam.SigmaSq);
    otherwise % user defined
        y = DefaultKernel(x1, x2);
end

% linear kernel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = LinearKernel(x1, x2)
y=x1*x2';

% Polynomail kernel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = PolynomialKernel(x1, x2, PolyDegree)
% PolyDegree = 2;
y=x1*x2';
y=y.^PolyDegree;

% Gaussian kernel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = GaussianKernel(x1, x2, SigmaSq)
% SigmaSq = 0.1;
x1NormSq = sum(x1.^2,2)';
x2NormSq = sum(x2.^2,2)';
InnerProduct = x1*x2';
y = abs(x1NormSq+x2NormSq-2*InnerProduct);
y = exp(-y/SigmaSq);

% Default kernel: linear
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = DefaultKernel(x1, x2)
y=x1*x2';
