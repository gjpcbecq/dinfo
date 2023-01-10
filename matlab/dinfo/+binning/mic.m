function [micXYKZ, hXZ, hYZ, hXYZ, hZ] = mic(x, y, z, varargin)
% Compute conditional mutual information h(x; y | z)
% 
% Syntax
%
% [micXYKZ, hXZ, hYZ, hXYZ, hZ] = mic(x, y, z, nBin=2, mode="marginal")
% 
% Input
%
% x is nDimX-by-nObs
% y is nDimY-by-nObs
% z is nDimZ-by-nObs
%
% Output
% 
% micXYKZ: float
% hXZ: float
% hYZ: float
% hXYZ: float
% hZ: float
% 
% Description
%
% mi(x; y | z) = h(x, z) + h(y, z) - h(x, y, z) - h(z)
% 
% Example
% 
% rng(1)
% [x, y, z] = model.GaussianXYZ(100, 0.9, 0.5, 0.1); 
% [micXYKZ, hXZ, hYZ, hXYZ, hZ] = binning.mic(x, y, z); 
% disp([micXYKZ, hXZ, hYZ, hXYZ, hZ]); 
% 
%   0.2069    3.0252    3.0228    4.2668    1.5742
%

if (nargin == 3)
    nBin = 2; 
    mode = 'marginal'; 
end
if (nargin == 4)
    nBin = varargin{1}; 
    mode = 'marginal'; 
end
if (nargin == 5)
    nBin = varargin{1}; 
    mode = varargin{2}; 
end

nObsX = size(x, 2); 
nObsY = size(y, 2); 
nObsZ = size(z, 2); 
if ((nObsX ~= nObsY) || (nObsX ~= nObsZ)), 
    micXYKZ = []; hXZ = []; hYZ = []; hXYZ = []; hZ = []; 
    return; 
end
xyz = cat(1, x, cat(1, y, z)); 
yz = cat(1, y, z); 
xz = cat(1, x, z);
hXYZ = binning.h(xyz, nBin, mode);
hYZ = binning.h(yz, nBin, mode);
hXZ = binning.h(xz, nBin, mode);
hZ = binning.h(z, nBin, mode);
micXYKZ = hXZ + hYZ - hXYZ - hZ; 
return 