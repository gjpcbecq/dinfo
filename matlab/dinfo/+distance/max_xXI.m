function dist = max_xXI(x, xI) 
% Compute all the distances of samples x to sample xi using max distance
% 
% Syntax
%
% dist = max_xXI(x, xI) 
%
% Input
%
% x: nDim-by-nObs, an array of observations
% xI: nDim-by-1, the observation to evaluate
%
% Output
% 
% dist: 1-by-nObs
%
% Example
% 
% x = [1, 2, 3, 4; 11, 12, 13, 14]; 
% xI = [5; 15]; 
% disp(distance.max_xXI(x, xI)); 
%
%   4     3     2     1
% 
nObs = size(x, 2); 
xC = (x - repmat(xI, 1, nObs));
dist = max(abs(xC), [], 1); 
return 

