function alpha = optimalKernelLearn(xL, wL, kernelMethod, lambda, param)
% Learn optimal parameters for kernel. 
% 
% Syntax
%
% alpha = optimalKernelLearn(xL, wL, kernelMethod, lambda, param)
%
% Input
% 
% xL: 1-by-nObsL nObsL, size of the learning set
% wL: nDimW-by-nObsL, predictor learning set
% lambda: float, optimization parameter
% param: 1-by-nParam, kernel parameters
% 
% Output 
% 
% alpha: (nObsL, 1) 
% 
% Description
%
% Find the optimal weights for the learning targets xL and the learning
% predictors wL given lambda and a kernel with its parameters. 
%
% Example
%
% xL = [1, 2, 3]; 
% wL = [2, 3, 4; 1, 1, 1];
% alpha = kernel.learnOptimalGaussianKernel(xL, wL, 1., 0.1); 
% disp(alpha); 
% 
%    0.6016
%    0.7974
%    2.4506
%
switch kernelMethod
    case {'Gaussian'}; 
        funGram = @kernel.GaussianGram;
    case {'linear'}; 
        funGram = @kernel.LinearGram;
    otherwise
        disp('unknown kernel... check parameters'); 
        alpha = nan; 
        return  
end
    
K = funGram(wL, param);
nObsL = size(xL, 2);
One = eye(nObsL);
xL = xL';
alpha = (K + lambda * One) \ xL;
% other solution with cholesky
%{
M = (K + lambda * One); 
L = chol(M, 'inf'); 
alpha = L' \ (L \ xL); 
%}
return 