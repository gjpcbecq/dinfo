function [logP, mT, vT, VT] = logP_linear(xL, xP, xT, sigma)
% Compute log-evidence for Gaussian process regression with linear kernel
% 
% Syntax
% 
% [logP, mT, vT, VT] = logP_linear(xL, xP, xT, sigma)
% 
% Input
% 
% xL: nDimXL-by-nObsL, learning set
% xP: nDimXP-by-nObsP, prediction or target values
% xT: nDimXL-by-nObsL, test set
% sigma: hyperparameter
%
% Output
% 
% logP: log evidence
% mT: predictive mean for test value
% vT: predictive covariance matrix for test value 
% VT: a posteriori covariance matrix
% 
% Description
% 
% $$ logP = a + b + c $$ 
% with 
% $$ a = - 1 / 2 \\, xP^t \\, (K^2 + sigma \\, I)^{-1} \\, xP $$
% $$ b = - \\sum \\log{(diag(K^2 + \\sigma \\, I)} $$
% $$ c = - 1 / 2 \\, n \\, \\log{(2.0 \\, \\pi)} $$
%
% Example 
% 
% xL = [1., 2., 3.]; 
% xP = [2., 3., 4.];
% xT = [0., 2., 3., 5.];
% listSigma = 0.1; 
% [logP, mT, vT, VT] = gpr.logP_linear(xL, xP, xT, listSigma);
% disp(logP); 
% disp(mT); 
% disp(vT); 
% disp(VT); 
%
%   -21.9198
% 
%          0
%     2.8551
%     4.2827
%     7.1378
% 
%          0    1.9901    2.9851    4.9752
%          0    0.1778    0.2667    0.4445
%          0    0.0716    0.1074    0.1790
% 
%          0         0         0         0
%          0    0.0029    0.0043    0.0071
%          0    0.0043    0.0064    0.0107
%          0    0.0071    0.0107    0.0178
% 
n = size(xL, 2);
s2 = sigma ^ 2; 
G_LL = kernel.LinearGram(xL); 
I = eye(n); 
M = G_LL + s2 * I; 
% f = xP
% f^t \, M^{-1} \, f 
% fx = M^{-1} \, f = (L \, L^t)^{-1} \, f 
% fx = (L^t)^{-1} \, L^{-1} \, f 
% fx = L^t \ (L \ f)
% invMDotXP = fx
L = chol(M, 'lower'); 
% invMDotXP = solve(L.T, solve(L, xP))
invMDotXP = M \ xP'; 
a = - 1 / 2 * xP * invMDotXP; 
b = - sum(log(diag(L))); 
c = - 1 / 2 * n * log(2 * pi);  
logP = a + b + c; 
[G_TL, G_LT]= kernel.LinearGramXY(xT, xL); 
G_TT = kernel.LinearGram(xT);
mT = G_TL * invMDotXP;
vT = L \ G_LT;
VT = G_TT - vT' * vT;
return 