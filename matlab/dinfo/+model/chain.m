function [x, y, z, e, M, M2] = chain(nObs)
% Simulate samples from the example 5.1. of Amblard et al. 2012 
% Workshop MLSP
%
% Syntax
%
% [x, y, z, e, M, M2] = chain(nObs)
%
% Input
% 
% nObs: number of samples to simulate
% 
% Output
% 
% x: 1-by-nObs
% y: 1-by-nObs
% z: 1-by-nObs
% e: 3-by-nObs, the noise
% M: 3-by-3, matrix of linear relations
% M2: 3-by-3, matrix of squared relations
%
% Description
% 
% x -> y -> z
%
% $$
% \left \{
% \begin{array}{l}
% x_t = a \, x_{t - 1} + \epsilon_{x, t} \\
% y_t = b \, y_{t - 1} + d_{xy} \, x_{t - 1} ^2 + \epsilon_{y, t} \\
% z_t = c \, z_{t - 1} + c_{yz} \, y_{t - 1} + \epsilon_{z, t} \\
% (a, b, c, d_{xy}, c_{yz}) = (0.2, 0.5, 0.8, 0.8, 0.7)
% \end{array}
% \right .
% $$
% Initial parameters are set to zeros.
% 
% Example
% 
% rng(1)
% nObs = 100;
% [x, y, z, e, M, M2] = model.chain(nObs);
% listLambda = [0.01, 0.1, 1]; 
% param = {[0.1, 1, 10]}; 
% nTrial = 10; 
% dXY = zeros(1, nTrial); 
% dXYKZ = zeros(1, nTrial); 
% for i = 1 : nTrial, 
%   [x, y, z, e, M, M2] = model.chain(nObs); 
%   dXY(i) = dinfo.gcm(x, y, z, 3, '', 'dynamic', 'Gaussian', 2, ...
%       listLambda, param);
%   dXYKZ(i) = dinfo.gcm(x, y, z, 3, 'conditional', 'dynamic', ...
%       'Gaussian', 2, listLambda, param);
%   disp([dXY(i), dXYKZ(i)]); 
% end
% [pXY, stepX, thX, xS] = binning.prob(dXY, 10);
% mXY = (thX(1:end-1) + thX(2:end)) / 2;
% [pXYKZ, stepX, thX, xS] = binning.prob(dXYKZ, 10);
% mXYKZ = (thX(1:end-1) + thX(2:end)) / 2;
% plot(mXY, pXY)
% hold on 
% plot(mXYKZ, pXYKZ, 'r')
%
%    -0.1098   -0.0071
%     0.0982    0.0252
%    -0.0128   -0.0073
%    -0.1433   -0.0427
%    -0.0632   -0.0076
%    -0.1027   -0.0133
%     0.2646   -0.0574
%     0.2921    0.0390
%     0.4078   -0.0248
%    -0.3065   -0.0169
%
 

IX = 1 ; IY = 2 ; IZ = 3; 
nDim = 3 ; 
X = zeros(nDim, nObs + 1); 
M = zeros(nDim, nDim); 
M2 = zeros(nDim, nDim); 
M(IX, IX) = 0.2 ; M(IY, IY) = 0.8 ; M(IZ, IZ) = 0.8 ; 
M2(IY, IX) = 0.8 ; M(IZ, IY) = 0.7; 
E = 1 * randn(nDim, nObs + 1); 
for p = 2 : nObs + 1, 
    pm1 = p - 1; 
    X(IX, p) = M(IX, IX) * X(IX, pm1) + E(IX, p); 
    X(IY, p) = (M(IY, IY) * X(IY, pm1) + M2(IY, IX) * (X(IX, pm1) ^ 2) ...
        + E(IY, p)); 
    X(IZ, p) = (M(IZ, IZ) * X(IZ, pm1) + M(IZ, IY) * X(IY, pm1) ...
        + E(IZ, p)); 
end
x = X(IX, 2:end); y = X(IY, 2:end); z = X(IZ, 2:end); e = E(:, 2:end); 
return 