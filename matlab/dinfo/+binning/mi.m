function [miXY, hX, hY, hXY] = mi(x, y, varargin) 
% Compute mutual information mi(x; y)
% 
% Syntax
%
% [miXY, hX, hY, hXY] = mi(x, y, nBin=2, mode='marginal')
% 
% Input
%
% x: nDimX-by-nObs
% y: nDimY-by-nObs
% nBin=2: int, number of bins
% mode='marginal': string, 'marginal' or 'total'
% 
% Output
%
% miXY: float 
% hX: float
% hY: float
% hXY: float
% 
% Description
% 
% mi(x; y) = h(x) + h(y) - h(x, y)
% 
% Example
%
% rng(1)
% x = 3 * rand(1, 100);
% y = 0.5 * rand(1, 100);
% [miXY, hX, hY, hXY] = binning.mi(x, y, 10); 
% disp([miXY, hX, hY, hXY]); 
%
%   0.4115    1.0640   -0.7956   -0.1430
%
% Example
%
% rng(1)
% x = 3 * rand(3, 100); 
% y = 1/2 * rand(2, 100); 
% [miXY, hX, hY, hXY] = binning.mi(x, y, 10); 
% disp([miXY, hX, hY, hXY]); 
% disp([0, 3 * log(3), 2 * log(0.5), 3 * log(3) + 2 * log(0.5)])
%
%   4.0574    0.8843   -1.9159   -5.0890
%        0    3.2958   -1.3863    1.9095
%
% Example
% 
% rng(1); 
% [x, y, z] = model.GaussianXYZ(100, 0.9, 0.5, 0.1); 
% [miXY, hX, hY, hXY] = binning.mi(x, y); 
% disp([miXY, hX, hY, hXY]); 
% 
%   0.1729    1.4511    1.5059    2.7841
%
% See also binning.h
%
if (nargin == 2)
    nBin = 2; 
    mode = 'marginal'; 
end
if (nargin == 3)
    nBin = varargin{1}; 
    mode = 'marginal'; 
end
if (nargin == 4)
    nBin = varargin{1}; 
    mode = varargin{2};  
end
nObsX = size(x, 2); 
nObsY = size(y, 2); 
if (nObsX ~= nObsY), 
    miXY = []; hX=[]; hY=[]; hXY=[]; 
    return; 
end
hX = binning.h(x, nBin, mode);
hY = binning.h(y, nBin, mode);
xy = [x; y]; 
hXY = binning.h(xy, nBin, mode);
miXY = hX + hY - hXY;
return
