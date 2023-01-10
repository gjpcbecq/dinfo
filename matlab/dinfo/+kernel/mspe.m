function [mspe, xTHat] = mspe(xL, wL, xT, wT, kernelMethod, lambda, param)
% Compute the mean squared prediction error with kernel methods 
%
% Syntax
% 
% [mspe, xTHat] = mspe(xL, wL, xT, wT, kernelMethod, lambda, param)
%
% Input
% 
% xL: 1-by-nObsL, learning set targets
% wL: nDimW-by-nObsL, learning set predictors
% xT: 1-by-nObsT, test set targets
% wT: nDimW-by-nObsT, test set predictors
% kernelMethod: 'Gaussian' or 'linear'
% lambda: float, the optimization parameter
% param: array of kernel parameters
% 
% Output 
% 
% mspe: float, mean squared prediction error on the test set.
% xTHat: 1-by-nObsT, the prediction on the test set. 
% 
% Description
% 
% Learn on wL, xL
% Test on wT, xT
% w contains the predictors
% x contains the targets
%
% Example
%
% xL = [1.1 , 1.9, 3.1]; 
% wL = [1., 2., 3.; 11., 12., 13.]; 
% xT = [1.1 , 1.9, 3.1]; 
% wT = [1., 2., 3.; 11., 12., 13.]; 
% [mspe, xHat] = kernel.mspe(xL, wL, xT, wT, 'Gaussian', 0.1, 10.); 
% disp(mspe); 
% disp(xHat); 
% 
%   0.2302
%   1.5180    1.9898    2.3873
% 
% See also kernel.mspe_crossfold, kernel.optimalKernelLearn,
%   kernel.optimalKernelTest 
%
alpha = kernel.optimalKernelLearn(xL, wL, kernelMethod, lambda, param); 
xTHat = kernel.optimalKernelTest(wT, kernelMethod, wL, alpha, param); 
mspe = util.mse(xT, xTHat); 
return