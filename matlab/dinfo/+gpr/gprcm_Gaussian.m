function [dXY, resP2, resP1, optimalParamP2, optimalParamP1] = ...
    gprcm_Gaussian(x, y, p, listSigma, listBeta)
% Compute Gaussian process regression causality measure with Gaussian 
%  kernel
% 
% Syntax
%
% [dXY, resP2, resP1, optimalParamP2, optimalParamP1] = ...
%   gprcm_Gaussian(x, y, p, listSigma, listBeta)
%
% Input
%
% x: nDimX-by-nObs
% y: nDimY-by-nObs
% p: int, order of the model
% listSigma: 1-by-nSigma, list of sigma values to evaluate 
% listBeta: 1-by-nBeta, list of beta values to evaluate 
% 
% Output 
% 
% dXY: float, = max(logP2) - max(logP1) = max(log(P(fy | x, y))) - 
%  max(log(P(fy | y)))
% resP2: nSigma-by-nBeta, result of logP2 for parameters
% resP1: nSigma-by-nBeta, result of logP1 for parameters
% optimalParamP2: 1-by-2, [sigma2Optimal, beta2Optimal]
% optimalParamP1: 1-by-2, [sigma1Optimal, beta1Optimal]
% 
% Example
% 
% x = [1., 2., 3., 4., 5., 6., 7.]; 
% y = [2., 3., 4., 5., 6., 7., 8.]; 
% listSigma = [0.01, 0.1, 1.]; 
% listBeta = [0.1, 1, 10]; 
% [dXY, resP2, resP1, optimalParamP2, optimalParamP1] = ... 
%  gpr.gprcm_Gaussian(x, y, 3, listSigma, listBeta); 
% disp(dXY)
% disp(resP2)
% disp(resP1)
% disp(optimalParamP2)
% disp(optimalParamP1)
%
%     -0.8037
% 
%     -90.6673  -90.3513  -41.9499
%     -89.8343  -89.5245  -35.9634
%     -48.5620  -48.4829  -25.2788
% 
%     -90.6673  -84.7796  -43.3268
%     -89.8343  -84.0571  -34.9969
%     -48.5620  -47.0312  -24.4751
% 
%      1    10
% 
%      1    10
% 
% Example
%
% rng(1); 
% [x, y, z] = model.GaussianXYZ(100, 0.9, 0.5, 0.1); 
% listSigma = [0.01, 0.1, 1.]; 
% listBeta = [0.01, 0.1, 1.]; 
% [dXY, resP2, resP1, optimalParamP2, optimalParamP1] = ... 
%  gpr.gprcm_Gaussian(x, y, 3, listSigma, listBeta); 
% disp(dXY)
% disp(resP2)
% disp(resP1)
% disp(optimalParamP2)
% disp(optimalParamP1)
% 
%     -0.0028
% 
%     -133.7798 -133.7799 -168.6852
%     -133.8200 -133.8201 -164.7353
%     -145.0759 -145.0759 -144.6477
% 
%     1.0e+04 *
% 
%     -0.0134   -0.0134   -1.3445
%     -0.0134   -0.0134   -0.0768
%     -0.0145   -0.0145   -0.0141
% 
%     0.0100    0.0100
% 
%     0.0100    0.1000
%
VERBOSE = 0; 
yTM1 = util.getTM1(y, p);
z = [x; y];
zTM1 = util.getTM1(z, p); 
% vy = var(y); 
fy = y; 
nSigma = size(listSigma, 2); 
nBeta = size(listBeta, 2); 
% logP2
optimalParamP2 = nan(2, 1); 
resP2 = zeros(nSigma, nBeta); 
maxLogP2 = -inf;  
xL = zTM1(:, p + 1 : end); 
xP = fy(:, p + 1 : end); 
xT = xL; 
if VERBOSE, disp('logP2'); disp(xL); disp(xP); disp(xT); end
for iSigma = 1 : nSigma, 
    sigma = listSigma(iSigma); 
    for iBeta = 1 : nBeta, 
        beta = listBeta(iBeta); 
        [logP2, ~, ~, ~] = gpr.logP_Gaussian(xL, xP, xT, sigma, beta); 
        resP2(iSigma, iBeta) = logP2;
        disp([logP2, maxLogP2]); 
        if (logP2 > maxLogP2)
            maxLogP2 = logP2; 
            optimalParamP2 = [sigma, beta]; 
        end
    end
end
% logP1
optimalParamP1 = nan(1); 
resP1 = zeros(nSigma, nBeta); 
maxLogP1 = -inf;
xL = yTM1(:, p + 1 : end);
xP = fy(:, p + 1 : end);
xT = xL;
if VERBOSE, disp('logP1'); disp(xL); disp(xP); disp(xT); end
for iSigma = 1 : nSigma
    sigma = listSigma(iSigma); 
    for iBeta = 1 : nBeta
        beta = listBeta(iBeta); 
        [logP1, ~, ~, ~] = gpr.logP_Gaussian(xL, xP, xT, sigma, beta); 
        resP1(iSigma, iBeta) = logP1; 
        disp([logP1, maxLogP1]); 
        if (logP1 > maxLogP1)
            maxLogP1 = logP1; 
            optimalParamP1 = [sigma, beta]; 
        end
    end
end
dXY = maxLogP2 - maxLogP1; 
return 