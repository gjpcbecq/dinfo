function dXY = gcmd(x, y, p, kernelMethod, nFold, listLambda, param) 
% Compute Geweke's dynamic causal measure on y from x with kernel using
% n fold cross-validation
% 
% Syntax
%
% dXY = gcmd(x, y, p, kernelMethod, nFold, listLambda, param)
%
% Input 
% 
% x: nDimX-by-nObs
% y: 1-by-nObs
% p: int, order of the model
% kernelMethod: 'Gaussian', 'linear'
% nFold: int, number of parts in cross-validation
% listLambda: 1-by-nLambda, array of optimization parameter
% param: 1-by-nParam, cell array of arrays of kernel parameters
%
% Output
% 
% dXY: float
%
% Description
%
% $$ G_{x \rightarrow y} = \frac {\sigma^2(y_t | y^{t - 1})} 
%  {\sigma^2(y_t | y^{t - 1}, x^{t - 1})} $$ 
%
% Example
% 
% x = 1: 100; 
% y = 0: 99;
% listLambda = [0.01, 0.1, 1.]; 
% param = {[1, 10, 50]}; 
% disp(funGewekeKernel.gcmd(x, y, 1, 'Gaussian', 2, listLambda, param));
% disp(funGewekeKernel.gcmd(y, x, 1, 'Gaussian', 2, listLambda, param));
% disp(funGewekeKernel.gcmd(x, y, 1, 'linear', 2, listLambda, param));
% disp(funGewekeKernel.gcmd(y, x, 1, 'linear', 2, listLambda, param));
%
%    -0.7183
%    -0.7226
%    11.8189
%     9.5852
% 
% Example
%
% rng(1)
% listLambda = [0.01, 0.1, 1]; 
% param = {[0.1, 1, 10]}; 
% [x, y, z] = model.GaussianXYZ(100, 0.9, 0.5, 0.1); 
% disp(funGewekeKernel.gcmd(x, y, 1, 'Gaussian', 2, listLambda, param));
%
%   0.0223
% 
xtm1 = util.getTM1(x, p);
ytm1 = util.getTM1(y, p);
% num
w = ytm1;
num = kernel.mspe_crossfold_search(y(p + 1 : end), w(:, p + 1 : end), ...
    kernelMethod, nFold, listLambda, param);
% den
w = [ytm1 ; xtm1];
den = kernel.mspe_crossfold_search(y(p + 1 : end), w(:, p + 1 : end), ...
    kernelMethod, nFold, listLambda, param);
% final
dXY = num / den;
dXY = log(dXY);
return
