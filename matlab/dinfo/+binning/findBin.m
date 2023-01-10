function iBin = findBin(th, x)
% Find the index of the bin given a set of thresholds such that
% th(i) <= x < th(i + 1)
%
% Syntax
%
% iBin = findBin(th, x)
% 
% input
%
% th: 1-by-nTh, set of thresholds
% x: 1-by-nObs, array of values
%
% Output
% 
% iBin: 1-by-nObs, array of bin indices. 
%
% Description
%
% first bin is defined by x < th(i) returns 1
% last bin is defined by x >= th(nTh) returns nTh = numel(th)
% 
% Example
% 
% th = [1.5, 2., 3.]; 
% x = [0, 1.5, 1.6, 2., 2.5, 3, 3.1]; 
% iBin = binning.findBin(th, x); 
% disp(iBin)
%
%   0     1     1     2     2     3     3
% 
n = numel(x); 
iBin = zeros(1, n); 
nTh = numel(th); 
for i = 1 : n
    iBin(i) = findBinOneElem(x(i), th, nTh); 
end
return 

function iBin1 = findBinOneElem(xi, th, nTh)
iBin1 = 1; 
while 1  
    if (th(iBin1) <= xi)
        if (iBin1 == nTh)
            iBin1 = nTh + 1; 
            break
        else
            iBin1 = iBin1 + 1; 
        end
    else
        break 
    end
end
% return 1 for 1st bin containing data 
iBin1 = iBin1 - 1; 
return
