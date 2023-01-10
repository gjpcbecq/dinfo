function [dXY, resP2, resP1, optimalParamP2, optimalParamP1] = ...
    gprcm_linear(x, y, p, listSigma)
% Compute Gaussian process regression causality measure with linear kernel 
% 
% Syntax
%
% [dXY, resP2, resP1, optimalParamP2, optimalParamP1] = ...
%   gprcm_linear(x, y, p, listSigma)
%
% Input 
% 
% x: nDimX-by-nObs
% y: nDimY-by-nObs
% p: int, order of the model
% listSigma: 1-by-nSigma, list of sigma values to evaluate 
% 
% Output 
% 
% dXY: float = maxlog(P2) - maxlog(P1) = max(log(P(fy | x, y))) - 
%   max(log(P(fy | y)))
% resP2: nSigma-by-1 result of logP2 for parameters
% resP1: nSigma-by1 result of logP1 for parameters
% optimalParamP2: float, optimal value for P2 
% optimalParamP1: float, optimal value for P1
% 
% Example
% 
% x = [1., 2., 3., 4., 5., 6., 7.];
% y = [2., 3., 4., 5., 6., 7., 8.];
% listSigma = [0.01, 0.1, 1.];
% [dXY, resP2, resP1, optimalParamP2, optimalParamP1] = ...
%   gpr.gprcm_linear(x, y, 3, listSigma); 
% disp(dXY); 
% disp(resP2); 
% disp(resP1); 
% disp(optimalParamP2); 
% disp(optimalParamP1); 
%
%     -0.3373
% 
%     1.6369
%     -2.9685
%     -7.6472
% 
%     1.9742
%     -2.6238
%     -7.2415
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
%  gpr.gprcm_linear(x, y, 3, listSigma); 
% disp(dXY)
% disp(resP2)
% disp(resP1)
% disp(optimalParamP2)
% disp(optimalParamP1)
% 
%     -2.3182
% 
%     1.0e+05 *
% 
%     -3.6502
%     -0.0354
%     -0.0014
%     -0.0031
% 
%     1.0e+05 *
% 
%     -3.9027
%     -0.0379
%     -0.0013
%     -0.0031
% 
%      1
% 
%      1
% 
yTM1 = util.getTM1(y, p);
z = [x; y];
zTM1 = util.getTM1(z, p);
fy = y; 
% logP2
optimalParamP2 = nan(1); 
nSigma = size(listSigma, 2); 
resP2 = zeros(nSigma, 1); 
maxLogP2 = -inf; 
xL = zTM1(: , p + 1: end);
xP = fy(: , p + 1: end);
xT = xL;
for iSigma = 1 : nSigma, 
    sigma = listSigma(iSigma); 
    [logP2, ~, ~, ~] = gpr.logP_linear(xL, xP, xT, sigma); 
    resP2(iSigma) = logP2; 
    if (logP2 > maxLogP2)
        maxLogP2 = logP2; 
        optimalParamP2 = sigma; 
    end
end
% logP1
optimalParamP1 = nan(1); 
resP1 = zeros(nSigma, 1); 
maxLogP1 = -inf; 
xL = yTM1(: , p + 1: end);
xP = fy(: , p + 1: end);
xT = xL;
for iSigma = 1 : nSigma, 
    sigma = listSigma(iSigma); 
    [logP1, ~, ~, ~] = gpr.logP_linear(xL, xP, xT, sigma); 
    resP1(iSigma) = logP1; 
    if (logP1 > maxLogP1)
        maxLogP1 = logP1;
        optimalParamP1 = sigma;
    end
end
dXY = maxLogP2 - maxLogP1; 
return 