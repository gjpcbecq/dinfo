function [hcXKY, hXY, hY] = hc(x, y, varargin)
% Compute conditional entropy h(x | y)
%
% Syntax
% 
% [hcXKY, hXY, hY] = hc(x, y, nBin=2, mode='marginal')
% 
% Input 
% 
% x: nDimX-by-nObs
% y: nDimY-by-nObs
% nBin=2: int, number of bins
% mode="marginal": string, "marginal" or "total"
% 
% Output
% 
% hXKY: float 
% hXY: float
% hY: float
%
% Description
% 
% h(x | y) = h(x, y) - h(y)
% 
% Example
%
% rng(1);
% x = 3 * rand(3, 100);
% y = 0.5 * rand(2, 100);
% [hcXKY, hXY, hY] = binning.hc(x, y);
% disp([hcXKY, hXY, hY]);
% disp([3 * log(3), 3 * log(3) - 2 * log(2), - 2 * log(2)]);
%
%   3.0879    1.6614   -1.4265
%   3.2958    1.9095   -1.3863
%
% Example
% 
% rng(1)
% [x, y, z] = model.GaussianXYZ(100, 0.9, 0.5, 0.1); 
% disp(binning.hc(x, y, 10)); 
%
%   0.3854
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
    hcXKY = []; hY=[]; hXY=[]; 
    return; 
end
xy = cat(1, x, y);
hXY = binning.h(xy, nBin, mode);
hY = binning.h(y, nBin, mode);
hcXKY = hXY - hY ;
return 
