function miXY = mi(x, y, varargin)
% Compute mutual information of x and y, mi(x ; y)
% 
% Syntax
%
% miXY = mi(x, y, method='bin', param={2})
%
% Input
% 
% x: nDimX-by-nObs
% y: nDimY-by-nObs
% varargin: 
%     method='bin': {'bin', 'binning'}, {'Kraskov'}, {'Frenzel'}
%     param={2}: cell array of parameters
% 
% Output
% 
% miXY: float
% 
% Description
%
% $$ i(x ; y) $$
% wrapper to binning.mi, funKraskov.mi, funFrenzel.mi
%
% Example
% 
% rng(1)
% [x, y, z] = model.GaussianXYZ(100, 0.9, 0.5, 0.1); 
% disp(dinfo.mi(x, y));
% disp(dinfo.mi(x, y, 'bin', {10}));
% disp(dinfo.mi(x, y, 'Kraskov', {10})); 
% disp(dinfo.mi(x, y, 'Frenzel', {10, 'Euclidean'})); 
%
%   0.1729
%   0.9085
%   0.6940
%   0.6940
%    
% See also binning.mi, funKraskov.mi, funFrenzel.mi
% 
if (nargin == 2)
    method = 'bin'; 
    param = {2}; 
end
if (nargin == 3)
    method = varargin{1}; 
    param = {2}; 
end
if (nargin == 4)
    method = varargin{1}; 
    param = varargin{2}; 
end
    
switch method, 
    case {'bin', 'binning'}, 
        miXY = binning.mi(x, y, param{1}); 
    case {'Kraskov'}, 
        miXY = funKraskov.mi(x, y, param{1}); 
    case {'Frenzel'}, 
        miXY = funFrenzel.mi(x, y, param{1}, param{2}); 
    otherwise 
        disp('unknown method... check parameters'); 
        miXY = nan; 
end
return 