function dist = Euclid_xXI(x, xI) 
% Compute all the distances of samples x to sample xi using Euclidean
% distance
% 
% Syntax
%
% dist = Euclid_xXI(x, xI) 
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
% disp(distance.Euclid_xXI(x, xI)); 
% 
%     5.6569    4.2426    2.8284    1.4142
%
nObs = size(x, 2); 
xC = (x - repmat(xI, 1, nObs)); 
dist = sqrt(sum(xC.^2, 1)); 
return 