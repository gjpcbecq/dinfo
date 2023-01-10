function [GXY, GYX] = GaussianGramXY(X, Y, beta) 
% Compute a Gram matrix with Gaussian kernel from matrix X and Y
% 
% Syntax
% 
% [GXY, GYX] = GaussianGramXY(X, Y, beta) 
%
% Input 
% 
% X: nDim-by-nObsX
% Y: nDim-by-nObsY
% beta: kernel parameter of the Gaussian kernel
% 
% Output
% 
% GXY nObsX-by-nObsY
% GYX = GXY.T
% 
% Description
% 
% X contains the vector in column X_i = X(:, i). 
% Same for Y
% 
% Example
% 
% x = [1, 2, 3; 2, 3, 4]; 
% y = [1, 2, 3, 4, 5, 1; 2, 3, 4, 5, 6, 2]; 
% [GXY, GYX] = kernel.GaussianGramXY(x, y, 2); 
% disp(GXY);
% disp(GYX);
% 
%   1.0000    0.6065    0.1353    0.0111    0.0003    1.0000
%   0.6065    1.0000    0.6065    0.1353    0.0111    0.6065
%   0.1353    0.6065    1.0000    0.6065    0.1353    0.1353
% 
%   1.0000    0.6065    0.1353
%   0.6065    1.0000    0.6065
%   0.1353    0.6065    1.0000
%   0.0111    0.1353    0.6065
%   0.0003    0.0111    0.1353
%   1.0000    0.6065    0.1353
%    
nObsX = size(X, 2); 
nObsY = size(Y, 2); 
GXY = zeros(nObsX, nObsY); 
for i = 1 : nObsX 
    for j = 1 : nObsY 
        GXY(i, j) = kernel.kernelGaussian(X(:, i), Y(:, j), beta); 
    end
end
GYX = GXY'; 
return 