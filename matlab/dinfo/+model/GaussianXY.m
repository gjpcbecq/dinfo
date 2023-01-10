function [x, y] = GaussianXY(nObs, rho)
% Simulate samples from a Gaussian bivariate model X and Y
% 
% Syntax
% 
% [x, y] = GaussianXY(nObs, rho)
%
% Input
%
% nObs: float, number of observations
% rho: float, correlation coefficient
% 
% Output
% 
% x: 1-by-nObs
% y: 1-by-nObs
%
% Example
% 
% rng(1)
% [x, y] = model.GaussianXY(1000, 0.9); 
% disp(corrcoef(x', y')); 
%
% 1.0000    0.8981
% 0.8981    1.0000
%
X = randn(2, nObs);
C = [1, rho; rho, 1]; 
R = chol(C, 'lower'); 
Y = R * X; 
x = Y(1, :); 
y = Y(2, :); 
return 
