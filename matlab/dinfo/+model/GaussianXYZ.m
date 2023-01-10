function [x, y, z] = GaussianXYZ(nObs, rhoXY, rhoYZ, rhoZX) 
% Simulate samples from a Gaussian trivariate model X, Y and Z
% 
% Syntax 
%
% [x, y, z] = GaussianXYZ(nObs, rhoXY, rhoYZ, rhoZX)
% 
% Input
% 
% nObs: float, number of observations
% rhoXY: float, correlation coefficient
% rhoYZ: float, correlation coefficient
% rhoZX: float, correlation coefficient
% 
% Output
% 
% x: 1-by-nObs
% y: 1-by-nObs
% z: 1-by-nObs
% 
% Example
% 
% rng(1)
% [x, y, z] = model.GaussianXYZ(1000, 0.9, 0.8, 0.7) 
% disp(corrcoef([x', y', z']))
% 
% 1.0000    0.9053    0.6915
% 0.9053    1.0000    0.7884
% 0.6915    0.7884    1.0000
%
X = randn(3, nObs); 
C = [1, rhoXY, rhoZX ; rhoXY, 1, rhoYZ ; rhoZX, rhoYZ, 1]; 
R = chol(C, 'lower'); 
Y = R * X; 
x = Y(1, :); 
y = Y(2, :); 
z = Y(3, :);
return
