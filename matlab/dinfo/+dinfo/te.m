function dXY = te(x, y, p, varargin)
% Compute transfert entropy, I(Dx^p -> y^p)
% 
% Syntax
%
% dXY = te(x, y, p, method='bin', param={2})
%
% Input
% 
% x: nDimX-by-nObs
% y: nDimY-by-nObs
% p: order of the model 
% varargin: 
%     method='bin': {'bin', 'binning'}, {'Kraskov'}, {'Frenzel'}
%     param={2}: cell array of parameters
% 
% Output
% 
% dXY: float
%
% Description
%
% See Amblard, P. O., & Michel, O. (2014). Causal conditioning and 
% instantaneous coupling in causality graphs. Information Sciences.
% $$ I(Dx \\rightarrow y) \\approx 
%     I(x_{t-p}^{t-1}; y_t | y_{t-p}^{t-1}) $$    
% 
% Example
% 
% rng(1)
% [x, y, z] = model.GaussianXYZ(100, 0.9, 0.5, 0.1); 
% disp(dinfo.te(x, y, 2));
% disp(dinfo.te(x, y, 2, 'Frenzel', {10, 'Euclidean'})); 
%
%    0.0551
%    0.0054
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
w = yTM1;
dXY = dinfo.mic(xTM1(:, p+1:end), y(:, p+1:end), w(:, p+1:end), ...
    method, param); 
return     