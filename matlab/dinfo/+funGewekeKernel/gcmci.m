function dXY = gcmci(x, y, z, p, kernelMethod, nFold, listLambda, param)
% Compute Geweke's conditional instantaneous causal measure on y from x 
% with kernel using n fold cross-validation
% 
% Syntax
%
% dXY = gcmci(x, y, z, p, kernelMethod, nFold, listLambda, param)
%
% Input 
% 
% x: nDimX-by-nObs
% y: 1-by-nObs
% z: nDimZ-by-nObs
% p: order of the model
% listLambda: 1-by-nLambda, array of optimization parameter
% param: 1-by-nParam, cell array of arrays of kernel parameters
%
% Output
% 
% dXY: float
%
% Description
% 
% $$ G_{x . y || z} = \frac{\sigma^2(y_t | y^{t - 1}, x^{t - 1}, z^{t})} 
%  {\sigma^2(y_t | y^{t - 1}, x^{t}, z^{t})} $$
% 
% Example
%
% x = 1: 100; 
% y = 0: 99; 
% z = ones(1, 100); 
% listLambda = [0.01, 0.1, 1.]; 
% param = {[1, 10, 50]}; 
% disp(funGewekeKernel.gcmci(x, y, z, 1, 'Gaussian', 2, listLambda, param))
% disp(funGewekeKernel.gcmci(y, x, z, 1, 'Gaussian', 2, listLambda, param))
%
%   -0.3247
%   -0.3258
% 
% Example
% 
% rng(1)
% listLambda = [0.01, 0.1, 1]; 
% param = {[0.1, 1, 10]}; 
% [x, y, z] = model.GaussianXYZ(100, 0.9, 0.5, 0.1); 
% disp(funGewekeKernel.gcmci(x, y, z, 1, 'Gaussian', 2, listLambda, param));
%
%   3.5733
% 
xtm1 = util.getTM1(x, p);
ytm1 = util.getTM1(y, p);
% ztm1 = util.getTM1(z, p);
xt = util.getT(x, p);
zt = util.getT(z, p);
% num
w = [ytm1 ; xtm1 ; zt];
num = kernel.mspe_crossfold_search(y(p + 1 : end), w(:, p + 1 : end), ...
    kernelMethod, nFold, listLambda, param);
% den
w = [ytm1 ; xt ; zt];
den = kernel.mspe_crossfold_search(y(p + 1 : end), w(:, p + 1 : end), ...
    kernelMethod, nFold, listLambda, param);
% final
dXY = num / den;
dXY = log(dXY);
return 