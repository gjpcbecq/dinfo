function [w, x, y, z, e, M, M2] = fourDimensional(nObs)
% Simulate samples from the example 5.2. of Amblard et al. 
% 2012 Workshop MLSP
% 
% Syntax
%
% [w, x, y, z, e, M, M2] = fourDimensional(nObs)
%
% Input
%
% nObs: number of samples to simulate
% 
% Output
% 
% w: 1-by-nObs
% x: 1-by-nObs
% y: 1-by-nObs
% z: 1-by-nObs
% e: 4-by-nObs, the noise
% M: 4-by-4, matrix of linear relation
% M2: 4-by-4, matrix of squared relations
%
% Description
% 
% Initial parameters are set to zeros.
% Noise are $\epsilon_{w, t}$, $\epsilon_{x, t}$, $\epsilon_{y, t}$, $\epsilon_{z, t}$ with covariance given by:  
% \begin{equation*}
% \Gamma_{\epsilon} = 
% \left( 
% \begin{array}{cccc}
% 1 & \rho_1 & 0 & \rho_1 \, \rho_2 \\
% \rho_1 & 1 & 0 & \rho_2 \\
% 0 & 0 & 1 & \rho_3 \\
% \rho_1 \, \rho_2 & \rho_2 & \rho_3 & 1
% \end{array}
% \right)
% \end{equation*}
% with
% $(\rho_1, \rho_2, \rho_3) = (0.66, 0.55, 0.48)$
% 
% Th multivariate are given by: 
% \begin{equation*}
% \left \{
% \begin{array}{lclclclclcl}
% w_t & = & 0.2 \, w_{t - 1} & - & 0.2 \, x_{t - 1} ^2 & & & + & 0.3 \, z_{t - 1} & + & \epsilon_{w, t} \\
% x_t & = & & & 0.3 \, x_{t - 1} & & & + & 0.3 \, z_{t - 1} ^2 & + & \epsilon_{x, t} \\
% y_t & = & & & 0.8 \, x_{t - 1} - 0.5 \, x_{t - 1} ^2 & - & 0.8 \, y_{t - 1} & & & + & \epsilon_{y, t} \\
% z_t & = & 0.2 \, w_{t - 1} & & & & & - & 0.4 \, z_{t - 1} & + & \epsilon_{z, t}
% \end{array}
% \right .
% \end{equation*}
% 
% Reference
% 
% Amblard et al., ``Kernelizing Geweke's measures of Granger Causality'', workshop on machine learning for signal processing, 2012
%
% Example
%
% rng(1)
% nObs = 100;
% [w, x, y, z, e, M, M2] = model.fourDimensional(nObs);
% listLambda = [0.01, 0.1, 1]; 
% param = {[0.1, 1, 10]};
% dXY = dinfo.gcm(w, x, y, 3, '', 'dynamic', 'Gaussian', 10, ...
%   listLambda, param); 
% disp(dXY)
%
%   -0.0212
% 
IW = 1; IX = 2; IY = 3; IZ = 4; 
nDim = 4; 
X = zeros(nDim, nObs + 1); 
M = zeros(nDim, nDim); 
M2 = zeros(nDim, nDim); 
M(IW, IW) = 0.2; M(IW, IZ) = 0.3; M2(IW, IX) = -0.2; 
M(IX, IX) = 0.3; M2(IX, IZ) = 0.3; 
M(IY, IY) = -0.8; M(IY, IX) = 0.8; M2(IY, IX) = -0.5; 
M(IZ, IZ) = -0.4; M(IZ, IW) = 0.2; 
Gamma = zeros(nDim, nDim); 
rho1 = 0.66; rho2 = 0.55 ; rho3 = 0.48; 
Gamma(1, :) = [1, rho1, 0, rho1 * rho2]; 
Gamma(2, :) = [rho1, 1, 0, rho2]; 
Gamma(3, :) = [0, 0, 1, rho3]; 
Gamma(4, :) = [rho1 * rho2, rho2, rho3, 1]; 
% Cholesky factorization of Gamma for multidimensional Gaussian
R = chol(Gamma); 
E = 1 * randn(nDim, nObs + 1);
E = R' * E;
for p = 2 : nObs + 1, 
    pm1 = p - 1; 
    X(IW, p) = (M(IW, IW) * X(IW, pm1) + M(IW, IZ) * X(IZ, pm1) + ... 
        M2(IW, IX) * (X(IX, pm1) ^ 2) + E(IW, p)); 
    X(IX, p) = (M(IX, IX) * X(IX, pm1) + M2(IX, IZ) * ... 
        (X(IZ, pm1) ^ 2) + E(IX, p)); 
    X(IY, p) = (M(IY, IY) * X(IY, pm1) + M(IY, IX) * X(IX, pm1) + ... 
        M2(IY, IX) * (X(IX, pm1) ^ 2) + E(IY, p)); 
    X(IZ, p) = (M(IZ, IZ) * X(IZ, pm1) + M(IZ, IW) * X(IW, pm1) + ...
        E(IZ, p));
end
w = X(IW, 2:end); x = X(IX, 2:end); y = X(IY, 2:end); z = X(IZ, 2:end); 
e = E(:, 2:end); 
return 