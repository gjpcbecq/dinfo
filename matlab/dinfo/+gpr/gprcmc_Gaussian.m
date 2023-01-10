function [dXYkZ, resP2, resP1, optimalParamP2, optimalParamP1] = ...
    gprcmc_Gaussian(x, y, z, p, listSigma, listBeta)
% Compute Gaussian process regression causality measure conditionally to z with Gaussian kernel. 
% 
% Syntax
% 
% [dXYkZ, resP2, resP1, optimalParamP2, optimalParamP1] = ...
%   gprcmc_Gaussian(x, y, z, p, listSigma, listBeta)
%
% Input
% 
% x: nDimX-by-nObs
% y: nDimY-by-nObs
% z: nDimZ-by-nObs
% p: order of the model
% listSigma: 1-by-nSigma, list of sigma values to evaluate 
% listBeta: 1-by-nBeta, list of beta values to evaluate 
% 
% Output
% 
% dXYkZ: float = max(logP2) - max(logP1) = max(log(P(fy | x, y, z))) - 
%  max(log(P(fy | y, z)))
% logP2: nSigma-by-nBeta result of logP2 for parameters
% logP1: nSigma-by-nBeta result of logP1 for parameters
% optimalParamP2: 1-by-2, [sigma2Optimal, beta2Optimal]
% optimalParamP1: 1-by-2, [sigma1Optimal, beta1Optimal]
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
% listBeta = [0.1, 1, 10];
% [dXY, resP2, resP1, optimalParamP2, optimalParamP1] = ...
%   gpr.gprcmc_Gaussian(x, y, z, 3, listSigma, listBeta);
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
%   gpr.gprcmc_Gaussian(x, y, z, 3, listSigma, listBeta); 
% disp(dXY)
% disp(resP2)
% disp(resP1)
% disp(optimalParamP2)
% disp(optimalParamP1)
%
%     0
% 
%     -133.7798 -133.7798 -136.9674
%     -133.8200 -133.8200 -136.8649
%     -145.0759 -145.0759 -145.2419
% 
%     -133.7798 -133.8200 -145.0759
%     -133.7798 -133.8200 -145.0759
%     -137.2298 -137.0541 -144.7257
% 
%     0.0100    0.0100
% 
%     0.0100    0.0100
% 
xyz = [x; y; z];
xyzTM1 = util.getTM1(xyz, p);
yz = [y; z];
yzTM1 = util.getTM1(yz, p);
fy = y; 
% logP2
optimalParamP2 = nan(2, 1); 
nBeta = size(listBeta, 2); 
nSigma = size(listSigma, 2); 
resP2 = zeros(nSigma, nBeta); 
maxLogP2 = -inf; 
xL = xyzTM1(: , p + 1: end);
xP = fy(: , p + 1: end);
xT = xL;
for iSigma = 1 : nSigma,
    sigma = listSigma(iSigma); 
    for iBeta = 1 : nBeta, 
        beta = listBeta(iBeta); 
        [logP2, ~, ~, ~] = gpr.logP_Gaussian(xL, xP, xT, sigma, beta); 
        resP2(iSigma, iBeta) = logP2; 
        if (logP2 > maxLogP2), 
            maxLogP2 = logP2; 
            optimalParamP2 = [sigma, beta]; 
        end
    end
end
% logP1
optimalParamP1 = nan(2, 1); 
resP1 = zeros(nBeta, nSigma); 
maxLogP1 = -inf; 
xL = yzTM1(:, p + 1: end);
xP = fy(:, p + 1: end);
xT = xL;
for iSigma = 1 : nSigma, 
    sigma = listSigma(iSigma); 
    for iBeta = 1 : nBeta,
        beta = listBeta(iBeta); 
        [logP1, ~, ~, ~] = gpr.logP_Gaussian(xL, xP, xT, sigma, beta); 
        resP1(iSigma, iBeta) = logP1; 
        if (logP1 > maxLogP1), 
            maxLogP1 = logP1; 
            optimalParamP1 = [sigma, beta]; 
        end
    end
end
dXYkZ = maxLogP2 - maxLogP1; 
return 