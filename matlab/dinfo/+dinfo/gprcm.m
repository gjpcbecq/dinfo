function dXY = gprcm(x, y, p, varargin)
% Compute Gaussian process regression causality measure, gpr(x->y)
% 
% Syntax 
% 
% dXY = dinfo.gprcm(x, y, p, kernel="Gaussian", 1, {1})
% 
% Input
% 
% x: nDimX-by-nObs
% y: 1-by-nObs
% p: integer, order of the model 
% varargin: 
%     1 - kernel='Gaussian': 'Gaussian'; 'linear'
%     2 - listSigma=1.: array of Sigma values 
%     3 - param={1}: cell array of array of parameters 
% 
% Output
% 
% dXY: float
% 
% Description
%
% $$ d_{x \rightarrow y} = 
%   \mbox{max}_{\theta_2} \log{P_2(y(t) | x^{t-1}, y^{t-1})} - 
%   \mbox{max}_{\theta_1} \log{P_1(y(t) | y^{t-1})} $$
% wrapper to gpr.gprcm_Gaussian or gpr.gprcm_linear
%
% Example
% 
% x = [1., 2., 3., 4., 5., 6., 7.];
% y = [2., 3., 4., 5., 6., 7., 8.];
% listLambda = [0.01, 0.1, 1.];
% param = {[0.1, 1, 10]};
% dXY = dinfo.gprcm(x, y, 3, 'Gaussian', listLambda, param); 
% disp(dXY); 
% dXY = dinfo.gprcm(x, y, 3, 'linear', listLambda, param); 
% disp(dXY); 
% 
%   -0.8037
%   -0.3373 
%
% Example
%
% rng(1)
% listLambda = [0.01, 0.1, 1]; 
% param = {[0.1, 1, 10]}; 
% [x, y, z] = model.GaussianXYZ(100, 0.9, 0.5, 0.1); 
% disp(dinfo.gprcm(x, y, 2, 'Gaussian', listLambda, param));
%
%   -1.8673
%
% See also gpr.gprcm_Gaussian, gpr.gprcm_linear, dinfo.gprcmc
%
if (nargin == 3), 
    kernel = 'Gaussian'; 
    listSigma = 1; 
    param = {1, 1}; 
end
if (nargin == 4), 
    kernel = varargin{1};
    disp('check value of parameters set to {1, 1} by default'); 
    listSigma = 1; 
    param = {1}; 
end
if (nargin == 5), 
    kernel = varargin{1}; 
    listSigma = varargin{2}; 
    param = {1}; 
end
if (nargin == 6), 
    kernel = varargin{1}; 
    listSigma = varargin{2}; 
    param = varargin{3}; 
end

switch kernel, 
    case 'Gaussian', 
        dXY = gpr.gprcm_Gaussian(x, y, p, listSigma, param{1});
    case 'linear', 
        dXY = gpr.gprcm_linear(x, y, p, listSigma);
    otherwise
        dXY = nan; 
        disp('unknown kernel... check parameters'); 
end
return 