function dXYkZ = gprcmc(x, y, z, p, varargin)
% Compute Gaussian process regression causality measure conditionally to z, gpr(x->y|z)
%
% dXYkZ = dinfo.gprcm(x, y, z, p, kernel="Gaussian", listLambda = 1, 
%   param={1, })
% 
% Input
% 
% x: nDimX-by-nObs
% y: 1-by-nObs
% z: nDimZ-by-nObs
% p: integer, order of the model 
% varargin: 
%    1 - kernel='Gaussian': 'Gaussian'; 'linear'
%    2 - param={1., 1.}: cell array of parameters 
% 
% Output
% 
% dXY: float
%
% Description
%
% $$ d_{x \rightarrow y | z} = 
%   \mbox{max}_{\theta_2} \log{P_2(y(t) | x^{t-1}, y^{t-1}, z^{t-1})} - 
%   \mbox{max}_{\theta_1} \log{P_1(y(t) | y^{t-1}, z^{t-1})} $$
% wrapper to gpr.gprcmc_Gaussian, gpr.gprcmc_linear
%
% Example
% 
% x = [1., 2., 3., 4., 5., 6., 7.];
% y = [2., 3., 4., 5., 6., 7., 8.];
% z = [1., 1., 1., 1., 1., 1., 1.];
% listLambda = [0.01, 0.1, 1.];
% param = {[0.1, 1, 10]};
% dXYkZ = dinfo.gprcmc(x, y, z, 3, 'Gaussian', listLambda, param); 
% disp(dXYkZ); 
% dXYkZ = dinfo.gprcmc(x, y, z, 3, 'linear', listLambda, param); 
% disp(dXYkZ); 
%
%   -0.8037
%   -0.4962
% 
% rng(1)
% listLambda = [0.01, 0.1, 1]; 
% param = {[0.1, 1, 10]}; 
% [x, y, z] = model.GaussianXYZ(100, 0.9, 0.5, 0.1); 
% disp(dinfo.gprcmc(x, y, z, 2, 'Gaussian', listLambda, param));
%
%   -0.0018
% 
% See also gpr.gprcmc_Gaussian, gpr.gprcmc_linear, dinfo.gprcm
% 
if (nargin == 4), 
    kernel = 'Gaussian'; 
    listSigma = 1; 
    param = {1}; 
end
if (nargin == 5), 
    kernel = varargin{1};
    listSigma = 1; 
    param = {1, 1}; 
end
if (nargin == 6), 
    kernel = varargin{1}; 
    listSigma = varargin{2}; 
    param = varargin{2}; 
end
if (nargin == 7), 
    kernel = varargin{1}; 
    listSigma = varargin{2}; 
    param = varargin{3}; 
end

switch kernel, 
    case 'Gaussian', 
        dXYkZ = gpr.gprcmc_Gaussian(x, y, z, p, listSigma, param{1});
    case 'linear', 
        dXYkZ = gpr.gprcmc_linear(x, y, z, p, listSigma);
    otherwise
        dXYkZ = nan; 
        disp('unknown kernel... check parameters'); 
end
return 