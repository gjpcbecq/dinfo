function xTHat = optimalKernelTest(wT, kernelMethod, wL, alpha, param) 
% Test prediction with optimal parameters using kernels 
% 
% Syntax
%
% xTHat = optimalKernelTest(wT, kernelMethod, wL, alpha, param) 
%
% Input
% 
% wT: (nDimW, nObsT) tested predictors 
% kernelMethod: string 'Gaussian' or 'linear'
% wL: (nDimW, nObsL) learned predictors
% alpha: (nObsL, ) learned optimal weights
% param: 1-by-nParam, kernel parameters
% 
% Output 
% 
% xTHat: (nObsT, ) outputs, targets, predictions 
% 
% Description
%
% Test learned kernels with optimal weights
%
% Example
% 
% wL = [2., 3, 4; 1, 1, 1]; 
% alpha = [0.6; 0.8; 2.5]; 
% wT = [2., 3, 4; 1, 1, 1]; 
% xT = kernel.optimalKernelTest(wT, 'Gaussian', wL, alpha, 1.); 
% disp(xT); 
% wT = [1., 2, 3; 1, 1, 1];
% xT = kernel.optimalKernelTest(wT, 'Gaussian', wL, alpha, 1.);
% disp(xT); 
% wT = [2., 3, 4; 3, 2, 1];
% xT = kernel.optimalKernelTest(wT, 'Gaussian', wL, alpha, 1.);
% disp(xT); 
% 
%     0.9401    1.9404    2.8053
%     0.2357    0.9401    1.9404
%     0.0172    0.7138    2.8053
% 
switch kernelMethod
    case {'Gaussian'}; 
        funVectorKernel = @kernel.vectorGaussianKernel;
    case {'linear'}
        funVectorKernel = @kernel.vectorLinearKernel;
    otherwise
        xTHat = nan; 
        return
end

nObsT = size(wT, 2); 
xTHat = zeros(1, nObsT); 
for i = 1 : nObsT, 
    kw = funVectorKernel(wT(:, i), wL, param); 
    xTHatTemp = kw * alpha; 
    xTHat(i) = xTHatTemp; 
end
% other solution using the Gram Matrix with wT and wL
%{
kW = GramGaussianXY(wT, wL, beta); 
xTHat = kW' * alpha; 
%}
return 