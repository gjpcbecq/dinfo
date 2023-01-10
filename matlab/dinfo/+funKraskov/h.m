function hX = h(x, varargin) 
% Compute entropy h(x)
% 
% Syntax
%
% hX = h(x, k=1)
%
% Input
% 
% x: nDim-by-nObs. 
% varargin: 
%   1 k=1: int, number of neighbors 
% 
% Output
% 
% hX: float
% 
% Description
% 
% This is actually the Leonenko entropy estimator, see Kraskov 2004.
% $$ \hat{H}(X) = \psi(k) + \psi(N) + \log{c_d} + 
%     \frac{d}{N} \, \sum_{i = 1} ^{N} \log{\epsilon_i} $$
%
% Example
% 
% rng(1); 
% x = 2 * rand(1, 1000); 
% hX = funKraskov.h(x, 10); 
% disp(['h exp = ', num2str(hX / log(2)), ' bits'])
% disp('h theory = 1 bit')
% 
%   h exp: 0.97146 bits
%   h theoric = 1 bit
% 
% Example
% 
% rng(1); 
% x = 3 * rand(4, 1000); 
% hX = funKraskov.h(x, 10); 
% disp(['h exp = ', num2str(hX), ' nats'])
% nDim = 4; 
% hTh = nDim * log(3); 
% disp(['h theory = ', num2str(hTh), ' nats'])
% 
%   h exp = 4.7682 nats
%   h theory = 4.3944 nats% 
%
% Example
% 
% rng(1); 
% x = randn(3, 1000); 
% hX = funKraskov.h(x, 10); 
% disp(['h exp = ', num2str(hX), ' nats'])
% detC = 1; 
% nDim = 3; 
% hTh = 1 / 2 * log((2 * pi * exp(1)) ^ nDim); 
% disp(['h theory = ', num2str(hTh), ' nats'])
%
%   h exp = 4.2641 nats
%   h theory = 4.2568 nats%
%
[d, nObs] = size(x); 
if (nargin == 1)
    k = 1; 
    k = min(k, nObs); 
end
if (nargin == 2)
    k = varargin{1}; 
end
cD = pi ^ (d / 2) / gamma(1 +  d / 2); 
epsilon = kNN.dist(x, k); 
% see Eq 20 Kraskov2004
hX1 = - psi(k) + psi(nObs) + log(cD); 
hX2 = d / nObs * sum(log(epsilon(k, :))); 
hX = hX1 + hX2; 
return 