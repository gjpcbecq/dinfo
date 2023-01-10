function xT = getTList(x, t) 
% Compute the extension x_{t-t0, t-t1, ..., t-tNT} at time t
% 
% Syntax
% 
% xT = util.getTList(x, t)
% 
% Input
% 
% x: (nDim, nObs)
% t: list of delays with nT values
% 
% Output
% 
% xT: (nT * nDim, nObs)
% 
% Description
% 
% $$ \\vec{x}(n) = (x1(n), x2(n), ..., xd(n))^t $$
% $$ \\vec{xT}(n) = (x1(n-t0), ..., x1(n-tNT), x2(n-t0), ..., xd(n-tNT))^t $$
% This is a generalization of getT and getTM1
% getT(x, p) = getTList(x, range(0, p+1))
% getTM1(x, p) = getTList(x, range(1, p+1))
% 
% Example 
% 
% x = [1, 2, 3, 4, 5]; 
% disp(util.getTList(x, [1, 3])); 
% 
%   0     0     0     3     4
%   0     0     0     1     2
% 
% Example
% 
% x = [1, 2, 3, 4, 5, 6, 7; 11, 12, 13, 14, 15, 16, 17]; 
% disp(util.getTList(x, [0, 2, 4])); 
% 
%   0     0     0     0     5     6     7
%   0     0     0     0     3     4     5
%   0     0     0     0     1     2     3
%   0     0     0     0    15    16    17
%   0     0     0     0    13    14    15
%   0     0     0     0    11    12    13
% 
% See also util.getT, util.getTM1

[nDim, nObs] = size(x); 
nT = size(t, 2); 
xT = zeros(nT * nDim, nObs); 
iTMax = max(t); 
for iObs = iTMax : nObs - 1, 
    for iDim = 0 : nDim - 1,  
        for iT = 0 : nT - 1,  
            i1 = nT * iDim + iT; 
            i2 = iObs - t(iT + 1); 
            xT(i1 + 1, iObs + 1) = x(iDim + 1, i2 + 1); 
        end
    end
end
return