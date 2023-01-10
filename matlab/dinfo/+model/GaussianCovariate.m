function x = GaussianCovariate(nObs, m, C)
% Simulate samples from a Gaussian covariate model given the covariance matrix C
% 
% Syntax
% 
% x = GaussianCovariate(nObs, m, C)
%
% Input
%
% nObs: int, number of observations
% m: nDim-by-1
% C: nDim-by-nDim
% 
% Output
% 
% x: nDim-by-nObs
%
% Example
% 
% rng(1)
% C = [1, 0.9, 0.3; 0.9, 1, 0.5; 0.3, 0.5, 1]; 
% m = [0.; 3.; -5.]; 
% x = model.GaussianCovariate(100, m, C); 
% disp(dinfo.h(x, 'Leonenko', {3}))
% disp(model.GaussianH(C))
%
%  3.0927
%  3.1967

mSize = size(m, 1); 
nDim = mSize(1); 
x = randn(nDim, nObs);
% Cholesky factorization of V for multidim Gaussian
R = chol(C, 'lower'); 
x = repmat(m, 1, nObs) + R * x;  
return 
