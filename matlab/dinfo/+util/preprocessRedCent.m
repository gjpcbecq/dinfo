function xC = preprocessRedCent(x) 
% Reduce and center an array of vectors. 
% 
% Syntax
% 
% xC = util.preprocessRedCent(x)
% 
% Input
% 
% x: nDim-by-nObs
% 
% Output
% 
% xC: nDim-by-nObs
% 
% Example 
% 
% x = [1., 2., 3., 4., 5.]; 
% xC = util.preprocessRedCent(x); 
% disp(xC)
% 
%   -1.2649   -0.6325         0    0.6325    1.2649
% 
[nDim, nObs] = size(x); 
xC = zeros(nDim, nObs); 
for i = 1 : nDim 
    xMean = mean(x(i, :));
    xSTD = std(x(i, :)); 
    xC(i, :) = (x(i, :) - xMean) / xSTD; 
end
