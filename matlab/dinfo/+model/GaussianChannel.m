function [x, y, z] = GaussianChannel(nObs, P, N)
% Simulate x, y, and z from a Gaussian channel y = x + z 
% 
% Syntax
% 
% [x, y, z] = GaussianChannel(nObs, P, N)
% 
% Input
% 
% nObs: int, number of observations
% P: variance of X 
% N: variance of Z
% 
% Output
% 
% x: 1-by-nObs
% y: 1-by-nObs
% z: 1-by-nObs
% 
% Description
% 
% X, Y and Z are Gaussian variables. 
% Y is the output of the channel.  
%
% $X \sim N(0, P)$
% $Z \sim N(0, N)$
% $Y \sim N(0, P + N)$
% 
% $Y_i = X_i + Z_i$
% 
% h(x) = 1 / 2 \, \log{(2\, \pi \, mbox{e}) \, (P)}
% h(z) = 1 / 2 \, \log{(2\, \pi \, mbox{e}) \, (N)}
% h(y) = 1 / 2 \, \log{(2\, \pi \, mbox{e}) \, (P + N)}
% 
% P = 2, N = 1: 
% mi(X; Y) = 1 / 2 \, \log{1 + P / N} = 0.5493
% 
% see Cover and Thomas, p. 261+
% 
% Example 
% 
% rng(1)
% [x, y, z] = model.GaussianChannel(1000, 2, 1); 
% miXY = dinfo.mi(x, y, 'Kraskov', {3}); 
% disp(miXY); 
% 
%    0.5228
% 
x = sqrt(P) * randn(1, nObs);
z = sqrt(N) * randn(1, nObs);
y = x + z;
