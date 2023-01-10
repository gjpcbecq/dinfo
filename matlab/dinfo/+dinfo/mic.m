function miXYkZ = mic(x, y, z, varargin)
% Compute conditional mutual information, i(x ; y | z)
% 
% Syntax
%
% miXYkZ = mic(x, y, z, method='bin', param={2})
% 
% Input 
% 
% x: nDimX-by-nObs
% y: nDimY-by-nObs
% z: nDimZ-by-nObs
% varargin: 
%     method='bin': {'bin', 'binning'}, {'Frenzel'}
%     param={2}: cell array of parameters
% 
% Output
%
% miXYkZ: float
%
% Description
%
% $$ i(x ; y | z) $$
% wrapper to binning.mic, funFrenzel.mic
% 
% Example
% 
% rng(1)
% [x, y, z] = model.GaussianXYZ(100, 0.9, 0.5, 0.1); 
% disp(dinfo.mic(x, y, z));
% disp(dinfo.mic(x, y, z, 'Frenzel', {10, 'Euclidean'})); 
%
%   0.2069
%   0.8160
%
% See also binning.mic, funFrenzel.mic
%
if (nargin == 3)
    method = 'bin'; 
    param = {2}; 
end
if (nargin == 4)
    method = varargin{1}; 
    disp('check parameters if using Frenzel')
    param = {2}; 
end
if (nargin == 5)
    method = varargin{1}; 
    param = varargin{2}; 
end

switch method 
    case {'bin', 'binning'} 
        miXYkZ = binning.mic(x, y, z, param{1}); 
    case {'Frenzel'}
        miXYkZ = funFrenzel.mic(x, y, z, param{1}, param{2});
    otherwise
        miXYkZ = nan; 
end
return 