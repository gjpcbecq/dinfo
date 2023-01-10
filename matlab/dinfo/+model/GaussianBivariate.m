function x = GaussianBivariate(nObs, rho)
% Simulate samples from two Gaussian covariate model with a given correlation 
% coefficient rho.
% 
% Syntax
%
% x = GaussianBivariate(nObs, rho)
%
% Input
%
% nObs: int, number of observations
% rho: float, correlation value 
%
% Output
%
% x: 2-by-nObs
% 
% Example
% 
% rng(1); 
% x = model.GaussianBivariate(1000, 0.5); 
% plot(x(1, :), x(2, :), '.')
% shg
% disp(corrcoef(x')); 
% disp(dinfo.h(x, 'Leonenko', {3})); 
% disp(model.GaussianH([1, 0.5; 0.5, 1])); 
% 
%   1.0000    0.5041
%   0.5041    1.0000
%
%   2.7298
%
%   2.6940
%
x = randn(2, nObs);
V = [1, rho; rho, 1];
% Cholesky factorization of V for multidim Gaussian
R = chol(V, 'lower'); 
x = R * x; 
return

