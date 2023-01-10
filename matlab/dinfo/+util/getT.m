function xT = getT(x, p)
% Generate the extension $x^{t}$ given order p
%
% Syntax
% 
% xT = getT(x, p)
%
% Input
% 
% x: nDim-by-nObs
% p: integer, order of the extension 
%
% Output 
%
% xT: ((p + 1) * nDim)-by-(nObs)
% 
% Description
%
% $$ \vec{x}(n) = (x1(n), x2(n), ..., xd(n))^t $$
% $$ \vec{xT}(n, p) = (x1(n), ..., x1(n-p), x2(n), ..., xd(n-p))^t $$
%
% Example 
% 
% x = [1, 2, 3, 4, 5]; 
% disp(util.getT(x, 2))
% 
%    0     0     3     4     5
%    0     0     2     3     4
%    0     0     1     2     3
% 
% x = [1, 2, 3, 4, 5; 11, 12, 13, 14, 15]; 
% disp(util.getT(x, 2))
% 
%    0     0     3     4     5
%    0     0     2     3     4
%    0     0     1     2     3
%    0     0    13    14    15
%    0     0    12    13    14
%    0     0    11    12    13
% 
% See also util.getTM1
%
xSize = size(x); 
nDim = xSize(1); 
nObs = xSize(2); 
xT = zeros((p + 1) * nDim, nObs); 
for iObs = p : nObs - 1, 
    for iDim = 0 : nDim - 1,  
        for iP = 0 : p, 
            i1 = ((p + 1) * iDim) + iP; 
            i2 = iObs - iP; 
            xT(i1 + 1, iObs + 1) = x(iDim + 1, i2 + 1); 
        end
    end
end
return