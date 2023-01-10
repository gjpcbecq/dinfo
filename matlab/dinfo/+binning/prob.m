function [p, stepX, thX, xS] = prob(x, nBin)
% Compute probability: [-inf, th(2)[, [th(i), th(i+1)[, [th(nBin), inf[
%
% Syntax
% 
% [p, stepX, thX, xS] = prob(x, nBin)
%
% Input
%
% x: nDim-by-nObs, multivariate
% nBin: int, number of bins
%
% Output
%
% p: 1-by-(nBin^nDim), probability to be in bins
% stepX: nDim-by-1, width of the bins
% thX: nDim-by-(nBin + 1), thresholds
% xS: nDim-by-nObs, sorted values
%
% Description
% 
% For each dimension: 
% bin_0: -inf <= x < th(1)
% bin_1: th(1) <= x < th(2)
% ...
% bin_nBin: th(nBin+1) <= x < inf
% 
% th(k) = min(x) + (k - 1) * range(x) / nBin
% mid(k) = min(x) + (k - 1 + 1/2) * range(x) / nBin 
% mid(k) = (th(k) + th(k + 1)) / 2
% th(k) = mid(k) + 1/2 * range(x) / nBin
% 
% -inf < xmin=th(1) < mid(1) < ... 
%     ... < th(nBin) < mid(nBin) < th(nBin+1) = xmax < inf
% 
% Indices in p are obtained with j = xBin(iDim, iObs) * (nBin ** iDim)
%
% Example
% 
% rng(1);
% x = rand(1, 100);
% [p, stepX, thX, xS] = binning.prob(x, 3);
% disp(p)
% disp(stepX)
% disp(thX)
% 
%    0.3500    0.2900    0.3600
%    0.3296
%    0.0001    0.3297    0.6593    0.9889
% 
[nDim, nObs] = size(x); 

if (nBin == 1)
    p = 1; 
    stepX = zeros(nDim, 1); 
    for iDim = 1 : nDim 
        stepX(iDim) = range(x(iDim, :)); 
    end
    thX = []; 
    xS = []; 
    return; 
end

xS = zeros(nDim, nObs); 
nTh = nBin + 1; 
thX = zeros(nDim, nTh); 
p = zeros(1, nBin ^ nDim);
xBin = zeros(nDim, nObs);
for iDim = 1 : nDim 
    [~, xS(iDim, :)] = sort(x(iDim, :)); 
    minX = x(iDim, xS(iDim, 1));
    maxX = x(iDim, xS(iDim, end));
    thX(iDim, :) = linspace(minX, maxX, nBin + 1); 
    xBin(iDim, :) = binning.findBin(thX(iDim, 2:nBin), x(iDim, :)); 
end
for iObs = 1 : nObs 
    j = 0; 
    for iDim = 1 : nDim 
        % first bin start at 0 here
        j = j + xBin(iDim, iObs) * (nBin ^ (iDim - 1)); 
    end
    j = j + 1; 
    p(j) = p(j) + 1; 
end
stepX = zeros(nDim, 1);
for iDim = 1 : nDim
    stepX(iDim) = thX(iDim, 2) - thX(iDim, 1); 
end
p = p / nObs; 
return 
