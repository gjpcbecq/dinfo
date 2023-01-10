function hXkY = hc(x, y, varargin)
% Compute conditional entropy of x given y, h(x | y)
%
% Syntax
% 
% hXkY = hc(x, y, method='bin', param={2})
%
% Input
% 
% x: nDimX-by-nObs
% y: nDimY, nObs
% varargin: 
%   method='bin': {'bin', 'binning'}, {'Kraskov'}, {'Frenzel'}
%   param={2}: cell array of parameters
% 
% Output
% 
% hXkY: float
%
% Description
%
% $$ h(x | y) $$
% wrapper to binning.hc, funKraskov.hc
% 
% Example
% 
% rng(1)
% [x, y, z] = model.GaussianXYZ(100, 0.9, 0.5, 0.1); 
% disp(dinfo.hc(x, y));
% disp(dinfo.hc(x, y, 'Leonenko', {10, })); 
% 
%   1.2782
%   0.5501
% 
% See also binning.hc, funKraskov.hc
% 
if (nargin == 2), 
    method = 'bin'; 
    param = {2}; 
end
if (nargin == 3), 
    method = varargin{1}; 
    param = {2}; 
end
if (nargin == 4), 
    method = varargin{1}; 
    param = varargin{2}; 
end
switch method, 
    case {'bin', 'binning'} 
        hXkY = binning.hc(x, y, param{1}); 
    case {'Kraskov'}
        hXkY = funKraskov.hc(x, y, param{1}); 
    otherwise 
        disp('unknown method... check argument'); 
        hXkY = nan; 
end
return 