function [dXYkZ, resP2, resP1, optimalParamP2, optimalParamP1] = ...
    gprcmc_linear(x, y, z, p, listSigma) 
% Compute Gaussian process regression causality measure conditionally to z  with linear kernel
% 
% Syntax
%
% [dXYkZ, resP2, resP1, optimalParamP2, optimalParamP1] = ...
%     gprcmc_linear(x, y, z, p, listSigma) 
%
% Input
% 
% x: nDimX-by-nObs
% y: nDimY-by-nObs
% z: nDimZ-by-nObs
% p: int, order of the model
% listSigma: 1-by-nSigma, list of sigma values to evaluate 
% 
% Output
% 
% dXYkZ: float = max(logP2) - max(logP1) = max(log(P(fy | x, y, z))) - max(log(P(fy | y, z)))
% resP2: nSigma-by-1, result of logP2 for parameters
% resP1: nSigma-by-1, result of logP1 for parameters
% optimalParamP2: float, optimal sigma value for P2 
% optimalParamP1: float, optimal sigma value for P1
% 
% Description
% 
% Influence of x on y conditionally to z : x -> y | z  
% If x causes y, dXYkZ = 0
%
% Example 
% 
% x = [1., 2., 3., 4., 5., 6., 7.];
% y = [2., 3., 4., 5., 6., 7., 8.]; 
% z = [1., 1., 1., 1., 1., 1., 1.];  
% listSigma = [0.01, 0.1, 1.]; 
% [dXY, resP2, resP1, optimalParamP2, optimalParamP1] = ...
%   gpr.gprcmc_linear(x, y, z, 3, listSigma); 
% disp(dXY)
% disp(resP2)
% disp(resP1)
% disp(optimalParamP2)
% disp(optimalParamP1)
%
%    -0.4962
% 
%     1.6198
%    -2.9864
%    -7.6943
% 
%     2.1160
%    -2.4914
%    -7.3003
% 
%     0.0100
% 
%     0.0100
% 
% Example
%
% rng(1); 
% [x, y, z] = model.GaussianXYZ(100, 0.9, 0.5, 0.1); 
% listSigma = [0.01, 0.1, 1., 10.]; 
% [dXY, resP2, resP1, optimalParamP2, optimalParamP1] = ... 
%  gpr.gprcmc_linear(x, y, z, 3, listSigma); 
% disp(dXY)
% disp(resP2)
% disp(resP1)
% disp(optimalParamP2)
% disp(optimalParamP1)
%
%     -1.7415
% 
%     1.0e+05 *
% 
%     -3.5184
%     -0.0342
%     -0.0014
%     -0.0032
% 
%     1.0e+05 *
% 
%     -3.5987
%     -0.0349
%     -0.0014
%     -0.0031
% 
%      1
% 
%      1
%
%
xyz = [x; y; z];
xyzTM1 = util.getTM1(xyz, p); 
yz = [y; z];
yzTM1 = util.getTM1(yz, p); 
fy = y;
% logP2
optimalParamP2 = nan(1);
nSigma = size(listSigma, 2); 
resP2 = zeros(nSigma, 1); 
maxLogP2 = -inf; 
xL = xyzTM1(:, p + 1 : end); 
xP = fy(:, p + 1 : end);
xT = xL;
for iSigma = 1 : nSigma, 
    sigma = listSigma(iSigma); 
    [logP2, ~, ~, ~] = gpr.logP_linear(xL, xP, xT, sigma); 
    resP2(iSigma) = logP2; 
    if (logP2 > maxLogP2), 
        maxLogP2 = logP2;
        optimalParamP2 = sigma; 
    end
end
% logP1
optimalParamP1 = nan(1); 
resP1 = zeros(nSigma, 1); 
maxLogP1 = -inf; 
xL = yzTM1(:, p + 1 : end);
xP = fy(:, p + 1 : end);
xT = xL;
for iSigma = 1 : nSigma, 
    sigma = listSigma(iSigma); 
    [logP1, ~, ~, ~] = gpr.logP_linear(xL, xP, xT, sigma); 
    resP1(iSigma) = logP1; 
    if (logP1 > maxLogP1), 
        maxLogP1 = logP1; 
        optimalParamP1 = sigma; 
    end
end
dXYkZ = maxLogP2 - maxLogP1; 
return 