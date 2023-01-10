function xTM1 = getTM1(x, p)
% Generate the extension $x^{t-1}$ given order p
% 
% Syntax
%
% xTM1 = getTM1(x, p)
%
% Input
% 
% x: nDim-by-nObs
% p: integer, order of the extension 
%
% Output
%
% xTM1: (p * nDim)-by-(nObs)
%
% Description
%
% $$ \vec{x}(n) = (x1(n), x2(n), ..., xd(n))^t $$
% $$ \vec{xTM1}(n, p) = (x1(n - 1), ..., x1(n - p), x2(n - 1), ... 
% xd(n - p))^t $$
%
% Example 
% 
% x = [1, 2, 3, 4, 5]; 
% disp(util.getTM1(x, 2))
% 
%    0     0     2     3     4
%    0     0     1     2     3
% 
% x = [1, 2, 3, 4, 5 ; 11, 12, 13, 14, 15]; 
% disp(util.getTM1(x, 2))
% 
%    0     0     2     3     4
%    0     0     1     2     3
%    0     0    12    13    14
%    0     0    11    12    13
% 
% See also util.getT
% 
xSize = size(x); 
nDim = xSize(1); 
nObs = xSize(2); 
xTM1 = zeros(p * nDim, nObs - p); 
for iObs = p : nObs - 1,
    for iDim = 0 : (nDim - 1), 
        for iP = 0 : p - 1, 
            i1 = (p * iDim) + iP; 
            i2 = iObs - iP - 1; 
            xTM1(i1 + 1, iObs + 1) = x(iDim + 1, i2 + 1); 
        end
    end
end
return 