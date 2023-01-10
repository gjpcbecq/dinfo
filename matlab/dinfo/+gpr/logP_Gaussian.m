function [logP, mT, vT, VT] = logP_Gaussian(xL, xP, xT, sigma, beta)
% Compute log-evidence for Gaussian process regression with Gaussian kernel
% 
% Syntax
%
% [logP, mT, vT, VT] = logP_Gaussian(xL, xP, xT, sigma, beta)
% 
% Input
% 
% xL: nDimXL-by-nObsL, learning set
% xP: nDimXP-by-nObsL, prediction or target values
% xT: nDimXL-by-nObsL, test set
% sigma: hyperparameter of the Gaussian process
% beta: parameter of the Gaussian kernel
% 
% Output
% 
% logP: float
% mT: predictive mean for test value
% vT: predictive covariance matrix for test value 
% VT: a posteriori covariance matrix
% 
% Description
% 
% $$ logP = a + b + c $$
% with 
% $$ a = - 1 / 2 \, xP^t \, (K^2 + sigma \, I)^{-1} \, xP $$
% $$ b = - \sum \log{(diag(K^2 + \sigma \, I)} $$
% $$ c = - 1 / 2 \, n \, \log{(2.0 \, \pi)} $$
%
% Example 
% 
% xL = [1., 2., 3.];  
% xP = [2., 3., 4.];  
% xT = [0., 2., 3., 5.]; 
% sigma = 0.01; 
% beta = 10;
% [logP, mT, vT, VT] = gpr.logP_Gaussian(xL, xP, xT, sigma, beta);
% disp(logP);
% disp(mT);
% disp(vT);
% disp(VT);
%
%   -28.7568
% 
%     1.0208
%     3.0084
%     3.9931
%     5.6560
% 
%     0.9900    0.9900    0.9607    0.8521
%    -0.1366    0.1407    0.2752    0.4975
%     0.0102    0.0053    0.0341    0.1427
% 
%     0.0011   -0.0002    0.0000    0.0017
%    -0.0002    0.0001    0.0000   -0.0004
%     0.0000    0.0000    0.0001    0.0004
%     0.0017   -0.0004    0.0004    0.0061
%
VERBOSE = 0; 
n = size(xL, 2); 
s2 = sigma ^ 2; 
G_LL = kernel.GaussianGram(xL, beta); 
if VERBOSE, disp(G_LL); end
I = eye(n); 
M = G_LL + s2 * I; 
if VERBOSE, disp(M); end
% f = xP
% f^t \, M^{-1} \, f 
% fx = M^{-1} \, f = (L \, L^t)^{-1} \, f 
% fx = (L^t)^{-1} \, L^{-1} \, f 
% fx = L^t \ (L \ f)
% invMDotXP = fx
L = chol(M, 'lower'); 
invMDotXP = M \ xP'; 
if VERBOSE, disp('xP'); disp(xP); end; 
if VERBOSE, disp('invMDotXP'); disp(invMDotXP); end; 
a = - 1 / 2 * xP * invMDotXP; 
b = - sum(log(diag(L))); 
c = - 1 / 2 * n * log(2 * pi);  
if VERBOSE, disp(a); disp(b); disp(c); end
logP = a + b + c; 
[G_TL, G_LT] = kernel.GaussianGramXY(xT, xL, beta); 
G_TT = kernel.GaussianGram(xT, beta); 
mT = G_TL * invMDotXP;
vT = L \ G_LT;
VT = G_TT - vT' * vT;
return 