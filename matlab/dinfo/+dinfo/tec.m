function dXYkZ = tec(x, y, z, p, varargin)
% Compute conditional transfert entropy, I(Dx^p -> y^p || Dz^p)
% 
% Syntax
%
% dXYkZ = tec(x, y, z, p, model='bin', param={2})
%
% Input
% 
% x: nDimX-by-nObs
% y: nDimY-by-nObs
% z: nDimZ-by-nObs
% p: order of the model 
% varargin: 
%     method='bin': {'bin', 'binning'}, {'Kraskov'}, {'Frenzel'}
%     param={2}: cell array of parameters
% 
% Output
% 
% dXYkZ: float
%
% Description
% 
% See Amblard, P. O., & Michel, O. (2014). Causal conditioning and
% instantaneous coupling in causality graphs. Information Sciences.
% $$ I(Dx \\rightarrow y \\| Dz) \\approx 
%     I(x_{t-p}^{t-1}; y_t | y_{t-p}^{t-1}, z_{t-p}^{t-1}) $$ 
% 
% Example
% 
% rng(1)
% [x, y, z] = model.GaussianXYZ(100, 0.9, 0.5, 0.1); 
% disp(dinfo.tec(x, y, z, 2));
% disp(dinfo.tec(x, y, z, 2, 'Frenzel', {10, 'Euclidean'})); 
%
%    0.0855
%    0.0032
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
w = [yTM1; zTM1]; 
dXYkZ = dinfo.mic(xTM1(:, p+1:end), y(:, p+1:end), w(:, p+1:end), ...
    method, param); 
return