function hX = h(x, varargin)
% Compute entropy h(x)
% 
% Syntax 
%
% hX = h(x, nBin=3, mode='marginal')
% 
% Input
% 
% x: nDim-by-nObs
% varargin: 
%   nBin=3: int, number of bins
%   mode='marginal': nBin is considered for marginals
%   mode='total': nBin is the total number of bins. 
%     nBin marginal is adjusted to nBin = ceil(nBin ^ (1 / nDim))
% 
% Output
%
% hX: float, entropy of x
%
% Description
% 
% h is given in nats, divide by log(2) to have it in bits. 
% 
% Example
%
% rng(1);
% x = randn(3, 100);
% disp(['h exp: ', num2str(binning.h(x, 10))])
% disp(['h theory: ', num2str(model.GaussianH(eye(3)))])
%
%   h exp: 2.2201
%   h theory: 4.2568
%
% Example
% 
% rng(1); 
% [x, y, z] = model.GaussianXYZ(100, 0.9, 0.5, 0.1); 
% disp(binning.h(x)); 
% 
%   1.4511
%
if (nargin == 1)
    nBin = 2;
    mode = 'marginal';
end
if (nargin == 2)
    nBin = varargin{1};
    mode = 'marginal';
end
if (nargin == 3)
    nBin = varargin{1};
    mode = varargin{2};
end

nDim = size(x, 1);
switch mode 
    case 'total'
        nBin = floor(nBin ^ (1 / nDim)); 
        if (nBin < 1)
            nBin = 1; 
        end
    case 'marginal'
    otherwise
        disp('undefined mode... check parameters'); 
        hX = nan; 
        return 
end
[p, stepX, ~, ~] = binning.prob(x, nBin); 
vol_hypercube = prod(stepX); 
hX = 0; 
nP = numel(p); 
for iP = 1 : nP
    if (p(iP) > 0) 
        hX = hX - p(iP) * log(p(iP)); 
    end
end
hX = hX + log(vol_hypercube); 
return  
