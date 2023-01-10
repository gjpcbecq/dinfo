function [GXY, GYX] = LinearGramXY(X, Y, varargin)
% Compute a Gram matrix with linear kernel from matrix X and Y
% 
% Syntax
%
% [GXY, GYX] = LinearGramXY(X, Y, varargin)
%
% Input 
% 
% X: nDim-by-nObsX
% Y: nDim-by-nObsY
% varargin: 
%  param: unusued but necessary for compatibility with other functions
%   calling it
% 
% Output
%
% GXY: nObsX-by-nObsY, the Gram matrix 
% GYX: nObsY-by-nObsX, GYX = GXY'
% 
% Description
%
% X contains the vector in column X_i = X(:, i). 
% Same for Y
%
% Example
%
% x = [1, 2, 3; 11, 12, 13]; 
% y = [1, 2, 3, 4, 5, 1; 11, 12, 13, 14, 15, 11]; 
% [GXY, GYX] = kernel.LinearGramXY(x, y); 
% disp(GXY); 
% disp(GYX); 
% 
%    122   134   146   158   170   122
%    134   148   162   176   190   134
%    146   162   178   194   210   146
% 
%    122   134   146
%    134   148   162
%    146   162   178
%    158   176   194
%    170   190   210
%    122   134   146
% 
% See also kernel.LinearGram, kernel.kernelLinear
%
nObsX = size(X, 2); 
nObsY = size(Y, 2); 
GXY = zeros(nObsX, nObsY); 
for i = 1 : nObsX, 
    for j = 1 : nObsY,  
        GXY(i, j) = kernel.kernelLinear(X(:, i), Y(:, j)); 
    end
end
GYX = GXY'; 
return 