function [x, y, z] = chain2(nObs, epsX, epsY, epsZ, a)
% Simulate samples from the chain model used in Amblard et al., A Gaussian Process Regression..., 2012
% 
% Syntax
% 
% [x, y, z] = modelChain2(nObs, epsX, epsY, epsZ, a)
%
% Input
% 
% nObs: int, number of observations
% epsX: 1-by-nObs
% epsY: 1-by-nObs
% epsZ: 1-by-nObs
% a=1.8: float
% 
% Output
%
% x: 1-by-nObs
% y: 1-by-nObs
% z: 1-by-nObs
% 
% Description
% 
% x(0), y(0) and z(0) are set to 0.  
% 
% $$
% \\left \\{
% \begin{array}{l}
% x(n) = 1 - a \\, x(n-1)^2 + epsX(n) \\\\
% y(n) = 0.8 \\, (1 - a \\, y(n-1)^2) + 0.2 \\, (1 - a \\, x(n-1)^2) + epsY(n) \\\\
% z(n) = 0.8 \\, (1 - a \\, z(n-1)^2) + 0.2 \\, (1 - a \\, y(n-1)^2) + epsZ(n)
% \\right . 
% $$
%
% Example
% 
% rng(1)
% nObs = 100;
% epsX = randn(1, nObs) * 0.01;
% epsY = randn(1, nObs) * 0.01;
% epsZ = randn(1, nObs) * 0.01;
% listLambda = [0.01, 0.1, 1]; 
% param = {[0.1, 1, 10]}; 
% [x, y, z] = model.chain2(nObs, epsX, epsY, epsZ, 1.8);
% disp(dinfo.gprcm(x, z, 3, 'Gaussian', listLambda, param)); 
% disp(dinfo.gprcmc(x, z, y, 3, 'Gaussian', listLambda, param)); 
%
%   -50.8367
%   -91.7092
%
x = zeros(1, nObs + 1); 
y = zeros(1, nObs + 1);
z = zeros(1, nObs + 1);
for t = 2 : nObs + 1,  
    tm1 = t - 1 ; 
    x2 = x(tm1); x2 = x2 * x2; 
    y2 = y(tm1); y2 = y2 * y2; 
    z2 = z(tm1); z2 = z2 * z2; 
    x(t) = 1 - a * x2 + epsX(tm1); 
    y(t) = 0.8 * (1 - a * y2) + 0.2 * (1 - a * x2) + epsY(tm1);
    z(t) = 0.8 * (1 - a * z2) + 0.2 * (1 - a * y2) + epsZ(tm1);
end
x = x(2:end); y = y(2:end); z = z(2:end); 
return        