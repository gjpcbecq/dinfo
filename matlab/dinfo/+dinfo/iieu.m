function dXYkZ = iieu(x, y, z, p, varargin) 
% Compute instantaneous unconditional information exchange, I(x^p -> y^p || Dx^p, Dz^p)
%
% Syntax
%
% dXYkZ = iieu(x, y, z, p, method='bin', param={2}) 
%
% Input
% 
% x: nDimX-by-nObs
% y: nDimY-by-nObs
% z: nDimZ-by-nObs
% p: order of the model 
% varargin: 
%     method='bin': {'bin', 'binning'}, {'Kraskov'}, {'Frenzel'}
%     param={2}: int, float or cell array of parameters
% 
% Output
% 
% dXYkZ: float
% 
% Description
% 
% See Amblard, P. O., & Michel, O. (2014). Causal conditioning and
% instantaneous coupling in causality graphs. Information Sciences.
% $$ I(x \\rightarrow y \\| Dx, Dz) \\approx 
%     I(x_t; y_t | x_{t-p}^{t-1}, y_{t-p}^{t-1}, z_{t-p}^{t-1}) $$ 
%
% Example
% 
% rng(1)
% [x, y, z] = model.GaussianXYZ(100, 0.9, 0.5, 0.1); 
% disp(dinfo.iieu(x, y, z, 2));
% disp(dinfo.iieu(x, y, z, 2, 'Frenzel', {10, 'Euclidean'})); 
%
%    0.2175
%    0.0621
%
% See also dinfo.mic
% 
if (nargin == 4)
    method = 'bin'; 
    param = {2}; 
end
if (nargin == 5)
    method = varargin{1}; 
    param = {2}; 
end
if (nargin == 6)
    method = varargin{1}; 
    param = varargin{2}; 
end
xTM1 = util.getTM1(x, p);
yTM1 = util.getTM1(y, p);
zTM1 = util.getTM1(z, p);
w = [xTM1; yTM1; zTM1]; 
dXYkZ = dinfo.mic(x(:, p+1:end), y(:, p+1:end), w(:, p+1:end), ...
    method, param);
return     