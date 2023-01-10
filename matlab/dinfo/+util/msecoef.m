function E = msecoef(x)
% Compute the mean squared errors between vectors in x. 
% 
% Syntax
%
% E = util.msecoef(x)
% 
% Input
% 
% x: nDim-by-nObs
% 
% Output 
% 
% E: nDim-by-nDim
% 
% Example 
% 
% x = [1, 2, 3, 4; 11, 12, 13, 14; 21, 22, 23, 24; 1, 2, 3, 4]; 
% E = util.msecoef(x); 
% disp(E); 
%
%      0   100   400     0
%    100     0   100   100
%    400   100     0   400
%      0   100   400     0
%
% See also util.mse
% 
nDim = size(x, 1); 
E = zeros(nDim); 
for iDim = 1 : nDim
    E(iDim, iDim) = 0; 
    for jDim = iDim + 1 : nDim
        E(iDim, jDim) = util.mse(x(iDim, :), x(jDim, :)); 
    end
end
E = E + E'; 
return