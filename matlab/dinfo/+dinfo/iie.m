function dXY = iie(x, y, p, varargin)
% Compute instantaneous information exchange, I(x^p -> y^p || Dx^p)
% 
% Syntax
% 
% dXY = iie(x, y, p, method='bin', param={2})
%
% Input
% 
% x: nDimX-by-nObs
% y: nDimY-by-nObs
% p: order of the model 
% varargin: 
%     method='bin': {'bin', 'binning'}, {'Kraskov'}, {'Frenzel'}
%     param=2: cell array of parameters
% 
% Output
% 
% dXY: float
% 
% Description
% 
% See Amblard, P. O., & Michel, O. (2014). Causal conditioning and
% instantaneous coupling in causality graphs. Information Sciences.
% $$ I(x \\rightarrow y \\| Dx) \\approx 
%     I(x_t; y_t | x_{t-p}^{t-1}, y_{t-p}^{t-1}) $$ 
%
% Example
% 
% rng(1)
% [x, y, z] = model.GaussianXYZ(100, 0.9, 0.5, 0.1); 
% disp(dinfo.iie(x, y, 2));
% disp(dinfo.iie(x, y, 2, 'Frenzel', {10, 'Euclidean'})); 
%
%   0.2166
%   0.1912
%
% See also dinfo.mic
%
if (nargin == 3)
    method = 'bin'; 
    param = {2}; 
end
if (nargin == 4)
    method = varargin{1}; 
    param = {2}; 
end
if (nargin == 5)
    method = varargin{1}; 
    param = varargin{2}; 
end
xTM1 = util.getTM1(x, p);
yTM1 = util.getTM1(y, p);
w = [xTM1; yTM1];
dXY = dinfo.mic(x(:, p+1:end), y(:, p+1:end), w(:, p+1:end), ...
    method, param);
return 